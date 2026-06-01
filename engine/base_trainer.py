import math
import os

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils.evaluator import autocast_context, evaluate_model, move_to_device, to_train_metrics
from utils.validate import validate
from utils.network_factory import get_model


class BaseTrainerModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.logged_metrics = {}

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def log(self, name, value, **kwargs):
        if torch.is_tensor(value):
            value = value.detach()
            if value.numel() == 1:
                value = value.float().cpu().item()
        self.logged_metrics[name] = value

    def consume_logged_metrics(self):
        metrics = dict(self.logged_metrics)
        self.logged_metrics.clear()
        return metrics

    def clear_validation_outputs(self):
        for attr in ("validation_step_outputs_preds", "validation_step_outputs_gts"):
            outputs = getattr(self, attr, None)
            if outputs is not None:
                outputs.clear()

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]

    def fit(self, train_loader, val_loader=None, experiment_logger=None, run_name=None, log_dir=None):
        torch.set_float32_matmul_precision("high")

        train_conf = self.opt.train
        self.device = self.resolve_device(train_conf.get("gpu_ids", []))
        self.to(self.device)

        optimizer, scheduler = self.build_optimizer_and_scheduler()
        amp_dtype = self.resolve_amp_dtype(train_conf.get("precision", "16"), self.device)
        scaler = self.build_grad_scaler(self.device, amp_dtype)
        eval_predictor = train_conf.get("eval_predictor", self.opt.get("eval_pipeline", None))

        train_epochs = int(train_conf.train_epochs)
        check_val_every_n_epoch = int(train_conf.check_val_every_n_epoch)
        accumulation_steps = train_conf.get("gradient_accumulation_steps", 1)
        log_dir = log_dir or os.path.join("logs", run_name or self.opt.name)

        best_value, best_path, global_step = None, None, 0
        try:
            for epoch in range(1, train_epochs + 1):
                train_loss, global_step = self.train_one_epoch(
                    train_loader=train_loader,
                    optimizer=optimizer,
                    scaler=scaler,
                    amp_dtype=amp_dtype,
                    epoch=epoch,
                    epochs=train_epochs,
                    accumulation_steps=accumulation_steps,
                    global_step=global_step,
                )
                metrics = {"train_loss": train_loss, "lr": self.optimizer_lr(optimizer)}

                if val_loader is not None and epoch % check_val_every_n_epoch == 0:
                    metrics.update(
                        self.validate_one_epoch(
                            val_loader=val_loader,
                            amp_dtype=amp_dtype,
                            epoch=epoch,
                            epochs=train_epochs,
                            predictor=eval_predictor,
                        )
                    )

                self.step_scheduler(scheduler, metrics)
                self.log_epoch_metrics(experiment_logger, metrics, global_step, epoch)

                metric_key, metric_value, metric_mode = self.checkpoint_metric(metrics)
                if metric_key is not None and self.is_better_metric(metric_value, best_value, metric_mode):
                    if best_path is not None and os.path.exists(best_path):
                        os.remove(best_path)
                    best_value = metric_value
                    best_path = os.path.join(log_dir, f"epoch={epoch:02d}-{metric_key}={metric_value:.4f}.ckpt")
                    self.save_checkpoint(best_path, optimizer, scheduler, epoch, global_step, metrics)

                self.save_checkpoint(
                    os.path.join(log_dir, "last.ckpt"),
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    metrics,
                )
        finally:
            if experiment_logger is not None:
                experiment_logger.finish()

        return {"best_path": best_path, "best_value": best_value, "global_step": global_step}

    def train_one_epoch(
        self,
        train_loader,
        optimizer,
        scaler,
        amp_dtype,
        epoch,
        epochs,
        accumulation_steps,
        global_step,
    ):
        self.train()
        self.consume_logged_metrics()
        optimizer.zero_grad(set_to_none=True)

        accumulation_steps = max(1, int(accumulation_steps))
        total_loss, total_samples = 0.0, 0
        num_batches = len(train_loader)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", dynamic_ncols=True)

        for batch_idx, batch in enumerate(pbar, start=1):
            batch = move_to_device(batch, self.device)
            batch_size = self.get_batch_size(batch)

            with autocast_context(self.device, amp_dtype):
                loss = self.training_step(batch)
                if not torch.is_tensor(loss):
                    raise TypeError("training_step must return a torch.Tensor loss.")
                backward_loss = loss / accumulation_steps

            if scaler.is_enabled():
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            should_step = batch_idx % accumulation_steps == 0 or batch_idx == num_batches
            if should_step:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            loss_value = float(loss.detach().float().cpu())
            total_loss += loss_value * batch_size
            total_samples += batch_size
            avg_loss = total_loss / max(total_samples, 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{self.optimizer_lr(optimizer):.2e}")

        return total_loss / max(total_samples, 1), global_step

    def validate_one_epoch(self, val_loader, amp_dtype, epoch, epochs, predictor=None):
        eval_model = getattr(self, "model", self)
        try:
            result = evaluate_model(
                eval_model,
                val_loader,
                predictor=predictor,
                device=self.device,
                amp_dtype=amp_dtype,
                desc=f"Epoch {epoch}/{epochs} val",
            )
            return to_train_metrics(result)
        except ValueError:
            return self.validate_with_trainer(val_loader, amp_dtype, epoch, epochs)

    def validate_with_trainer(self, val_loader, amp_dtype, epoch, epochs):
        self.eval()
        self.clear_validation_outputs()
        self.consume_logged_metrics()

        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} val", dynamic_ncols=True)
        with torch.no_grad():
            for batch in pbar:
                batch = move_to_device(batch, self.device)
                with autocast_context(self.device, amp_dtype):
                    self.validation_step(batch)

        self.on_validation_epoch_end()
        return self.consume_logged_metrics()

    def build_optimizer_and_scheduler(self):
        configured = self.configure_optimizers()
        if isinstance(configured, tuple):
            optimizers, schedulers = configured
        else:
            optimizers, schedulers = configured, None

        optimizer = optimizers[0] if isinstance(optimizers, (list, tuple)) else optimizers
        if isinstance(schedulers, (list, tuple)):
            scheduler = schedulers[0] if schedulers else None
        else:
            scheduler = schedulers
        return optimizer, scheduler

    def step_scheduler(self, scheduler, metrics):
        if scheduler is None:
            return

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            monitor = metrics.get("val_ap_epoch", metrics.get("val_mfm_loss_epoch"))
            if monitor is not None:
                scheduler.step(monitor)
            return

        scheduler.step()

    def save_checkpoint(self, path, optimizer, scheduler, epoch, global_step, metrics):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "global_step": global_step,
            "state_dict": self.state_dict(),
            "optimizer_states": [optimizer.state_dict()],
            "lr_schedulers": [scheduler.state_dict()] if scheduler is not None else [],
            "metrics": metrics,
        }
        torch.save(checkpoint, path)

    def log_epoch_metrics(self, experiment_logger, metrics, global_step, epoch):
        payload = {"epoch": epoch, **metrics}
        if experiment_logger is not None:
            experiment_logger.log(payload, step=global_step)
        printable = ", ".join(
            f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
            for key, value in metrics.items()
        )
        tqdm.write(f"Epoch {epoch} | {printable}")

    @staticmethod
    def resolve_device(gpu_ids):
        gpu_ids = BaseTrainerModule.as_list(gpu_ids)
        if torch.cuda.is_available() and gpu_ids:
            if len(gpu_ids) > 1:
                print(
                    f"Manual training loop uses cuda:{gpu_ids[0]}; "
                    "multi-GPU launch should be handled by the outer framework."
                )
            device_id = int(gpu_ids[0])
            torch.cuda.set_device(device_id)
            return torch.device(f"cuda:{device_id}")
        return torch.device("cpu")

    @staticmethod
    def resolve_amp_dtype(precision, device):
        if device.type != "cuda":
            return None

        precision = str(precision).lower()
        if precision in {"16", "16-mixed", "fp16", "float16"}:
            return torch.float16
        if precision in {"bf16", "bf16-mixed", "bfloat16"}:
            if not torch.cuda.is_bf16_supported():
                print("bf16 precision requested but this CUDA device does not support bf16; using fp16.")
                return torch.float16
            return torch.bfloat16
        return None

    @staticmethod
    def build_grad_scaler(device, amp_dtype):
        enabled = device.type == "cuda" and amp_dtype == torch.float16
        try:
            return torch.amp.GradScaler(device.type, enabled=enabled)
        except (AttributeError, TypeError):
            return torch.cuda.amp.GradScaler(enabled=enabled)

    @staticmethod
    def optimizer_lr(optimizer):
        return optimizer.param_groups[0]["lr"]

    @staticmethod
    def get_batch_size(batch):
        if torch.is_tensor(batch):
            return batch.shape[0] if batch.ndim > 0 else 1
        if isinstance(batch, dict):
            for value in batch.values():
                return BaseTrainerModule.get_batch_size(value)
        if isinstance(batch, (tuple, list)) and batch:
            return BaseTrainerModule.get_batch_size(batch[0])
        return 1

    @staticmethod
    def as_list(value):
        if value is None:
            return []
        if isinstance(value, (str, bytes)):
            return [value]
        try:
            return list(value)
        except TypeError:
            return [value]

    @classmethod
    def checkpoint_metric(cls, metrics):
        for key, mode in (("val_ap_epoch", "max"), ("val_mfm_loss_epoch", "min")):
            value = cls.metric_as_float(metrics, key)
            if value is not None and math.isfinite(value):
                return key, value, mode
        return None, None, None

    @staticmethod
    def metric_as_float(metrics, key):
        value = metrics.get(key)
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().float().cpu().item()
        return float(value)

    @staticmethod
    def is_better_metric(value, best_value, mode):
        if best_value is None:
            return True
        if mode == "min":
            return value < best_value
        return value > best_value


class Trainer(BaseTrainerModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.criterion = nn.BCEWithLogitsLoss()

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)['logits']
        loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)['logits']
        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(
                torch.float32).sigmoid().flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        self.log('val_acc_epoch', acc, logger=True, sync_dist=True)
        self.log('val_ap_epoch', ap, logger=True, sync_dist=True)
        self.log('val_racc_epoch', r_acc, logger=True, sync_dist=True)
        self.log('val_facc_epoch', f_acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

