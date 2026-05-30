import os
import hydra
import argparse
import wandb
import datetime
from itertools import islice

import torch
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tqdm.auto import tqdm

import engine
import data
import networks
from utils.common import load_config_with_cli, archive_files, seed_everything



def build_dataloader(conf):
    train_datasets = []
    for sub_data in conf.datasets.train.source:
        for sub_set in sub_data.sub_sets:
            train_data = eval(sub_data.target)(sub_data.data_root, conf.datasets.train.trsf,
                                               subset=sub_set, split=sub_data.split)
            train_datasets.append(train_data)
    train_datasets = ConcatDataset(train_datasets)

    val_datasets = []
    for sub_data in conf.datasets.val.source:
        for sub_set in sub_data.sub_sets:
            val_data = eval(sub_data.target)(sub_data.data_root, conf.datasets.val.trsf,
                                          subset=sub_set, split=sub_data.split)
            val_datasets.append(val_data)
    val_datasets = ConcatDataset(val_datasets)


    train_loader = DataLoader(train_datasets, batch_size=conf.datasets.train.batch_size, shuffle=True,
                              num_workers=conf.datasets.train.loader_workers)
    val_loader = DataLoader(val_datasets, batch_size=conf.datasets.val.batch_size, shuffle=False,
                            num_workers=conf.datasets.val.loader_workers)
    return train_loader, val_loader


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def resolve_device(gpu_ids):
    gpu_ids = as_list(gpu_ids)
    if torch.cuda.is_available() and gpu_ids:
        if len(gpu_ids) > 1:
            print(f"Manual training loop uses cuda:{gpu_ids[0]}; multi-GPU launch should be handled by the outer framework.")
        device_id = int(gpu_ids[0])
        torch.cuda.set_device(device_id)
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def resolve_amp_dtype(precision, device):
    if device.type != "cuda":
        return None

    precision = str(precision).lower()
    if precision in {"16", "16-mixed", "fp16", "float16"}:
        return torch.float16
    if precision in {"bf16", "bf16-mixed", "bfloat16"}:
        return torch.bfloat16
    return None


def autocast_context(device, amp_dtype):
    return torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_dtype is not None,
    )


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {key: move_to_device(value, device) for key, value in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(value, device) for value in batch)
    if isinstance(batch, list):
        return [move_to_device(value, device) for value in batch]
    return batch


def get_batch_size(batch):
    if torch.is_tensor(batch):
        return batch.shape[0] if batch.ndim > 0 else 1
    if isinstance(batch, dict):
        for value in batch.values():
            return get_batch_size(value)
    if isinstance(batch, (tuple, list)) and batch:
        return get_batch_size(batch[0])
    return 1


def build_optimizer_and_scheduler(model):
    configured = model.configure_optimizers()
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


def build_grad_scaler(device, amp_dtype):
    enabled = device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def optimizer_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def run_sanity_check(model, val_loader, device, amp_dtype, num_steps):
    if num_steps <= 0:
        return

    model.eval()
    model.clear_validation_outputs()
    total = min(num_steps, len(val_loader))
    pbar = tqdm(islice(val_loader, num_steps), total=total, desc="Sanity check", dynamic_ncols=True, leave=False)
    with torch.no_grad():
        for batch in pbar:
            batch = move_to_device(batch, device)
            with autocast_context(device, amp_dtype):
                model.validation_step(batch)

    model.clear_validation_outputs()
    model.consume_logged_metrics()


def train_one_epoch(model, train_loader, optimizer, scaler, device, amp_dtype, epoch, epochs, accumulation_steps,
                    global_step):
    model.train()
    model.consume_logged_metrics()
    optimizer.zero_grad(set_to_none=True)

    accumulation_steps = max(1, int(accumulation_steps))
    total_loss, total_samples = 0.0, 0
    num_batches = len(train_loader)
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", dynamic_ncols=True)

    for batch_idx, batch in enumerate(pbar, start=1):
        batch = move_to_device(batch, device)
        batch_size = get_batch_size(batch)

        with autocast_context(device, amp_dtype):
            loss = model.training_step(batch)
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
        pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{optimizer_lr(optimizer):.2e}")

    return total_loss / max(total_samples, 1), global_step


def validate_one_epoch(model, val_loader, device, amp_dtype, epoch, epochs):
    model.eval()
    model.clear_validation_outputs()
    model.consume_logged_metrics()

    pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} val", dynamic_ncols=True)
    with torch.no_grad():
        for batch in pbar:
            batch = move_to_device(batch, device)
            with autocast_context(device, amp_dtype):
                model.validation_step(batch)

    model.on_validation_epoch_end()
    return model.consume_logged_metrics()


def step_scheduler(scheduler, metrics):
    if scheduler is None:
        return

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        monitor = metrics.get("val_ap_epoch")
        if monitor is not None:
            scheduler.step(monitor)
        return

    scheduler.step()


def save_checkpoint(path, model, optimizer, scheduler, epoch, global_step, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": model.state_dict(),
        "optimizer_states": [optimizer.state_dict()],
        "lr_schedulers": [scheduler.state_dict()] if scheduler is not None else [],
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def metric_as_float(metrics, key):
    value = metrics.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().float().cpu().item()
    return float(value)


def init_wandb(conf, run_name):
    try:
        import swanlab
        os.environ["WANDB_MODE"] = "offline"
        swanlab.sync_wandb()
    except ImportError:
        pass

    return wandb.init(
        name=run_name,
        project="DeepfakeDetection",
        job_type="train",
        group=conf.name,
    )


def log_metrics(wandb_run, metrics, global_step, epoch):
    payload = {"epoch": epoch, **metrics}
    wandb_run.log(payload, step=global_step)
    printable = ", ".join(
        f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
        for key, value in metrics.items()
    )
    tqdm.write(f"Epoch {epoch} | {printable}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--cfg', type=str, default=None, required=True)
    args, cfg_args = parser.parse_known_args()
    conf = load_config_with_cli(args.cfg, args_list=cfg_args)
    conf = hydra.utils.instantiate(conf)

    seed_everything(conf.train.seed)

    train_loader, val_loader = build_dataloader(conf)
    today_str = conf.name +"_"+ datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
    wandb_run = init_wandb(conf, today_str)

    if os.getenv("LOCAL_RANK", '0') == '0':
        archive_files(today_str, exclude_dirs=['logs', 'wandb', '.git', 'exp_results'])

    model = eval(conf.train.pipeline)(opt=conf)
    torch.set_float32_matmul_precision('high')
    device = resolve_device(conf.train.gpu_ids)
    model.to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(model)
    amp_dtype = resolve_amp_dtype(conf.train.get('precision', "16"), device)
    scaler = build_grad_scaler(device, amp_dtype)

    train_epochs = int(conf.train.train_epochs)
    check_val_every_n_epoch = int(conf.train.check_val_every_n_epoch)
    accumulation_steps = conf.train.get('gradient_accumulation_steps', 1)
    log_dir = os.path.join('logs', today_str)
    best_ap, best_path, global_step = None, None, 0

    run_sanity_check(model, val_loader, device, amp_dtype, int(conf.train.get('num_sanity_val_steps', 2)))

    for epoch in range(1, train_epochs + 1):
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_dtype=amp_dtype,
            epoch=epoch,
            epochs=train_epochs,
            accumulation_steps=accumulation_steps,
            global_step=global_step,
        )
        metrics = {"train_loss": train_loss, "lr": optimizer_lr(optimizer)}

        if epoch % check_val_every_n_epoch == 0:
            metrics.update(validate_one_epoch(model, val_loader, device, amp_dtype, epoch, train_epochs))

        step_scheduler(scheduler, metrics)
        log_metrics(wandb_run, metrics, global_step, epoch)

        current_ap = metric_as_float(metrics, "val_ap_epoch")
        if current_ap is not None and (best_ap is None or current_ap > best_ap):
            if best_path is not None and os.path.exists(best_path):
                os.remove(best_path)
            best_ap = current_ap
            best_path = os.path.join(log_dir, f"epoch={epoch:02d}-val_ap_epoch={current_ap:.4f}.ckpt")
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, global_step, metrics)

        save_checkpoint(os.path.join(log_dir, "last.ckpt"), model, optimizer, scheduler, epoch, global_step, metrics)

    wandb_run.finish()
