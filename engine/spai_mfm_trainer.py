import torch

from engine.base_trainer import BaseTrainerModule
from utils.network_factory import get_model


class Trainer_SPAIMFM(BaseTrainerModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_losses = []
        self.real_only = bool(opt.train.get("real_only", True))

    def _select_images(self, batch):
        x, y = batch
        if not self.real_only:
            return x

        real_mask = (y % 2) == 0
        if real_mask.any():
            return x[real_mask]
        return x

    def training_step(self, batch):
        x = self._select_images(batch)
        output = self.model(x)
        loss = output["loss"]
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x = self._select_images(batch)
        output = self.model(x)
        self.validation_step_outputs_losses.append(output["loss"].detach())

    def on_validation_epoch_end(self):
        if self.validation_step_outputs_losses:
            loss = torch.stack(self.validation_step_outputs_losses).mean()
            self.log("val_mfm_loss_epoch", loss, logger=True, sync_dist=True)
        self.validation_step_outputs_losses.clear()

    def clear_validation_outputs(self):
        super().clear_validation_outputs()
        self.validation_step_outputs_losses.clear()

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]
