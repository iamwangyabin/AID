import torch
import torch.nn.functional as F

from engine.base_trainer import Trainer


class Trainer_FIRE(Trainer):
    def __init__(self, opt):
        super().__init__(opt)
        train_conf = getattr(self.opt, "train", {})
        self.bce_weight = float(train_conf.get("bce_weight", 0.6))
        self.rec_weight = float(train_conf.get("rec_weight", 0.2))
        self.mask_weight = float(train_conf.get("mask_weight", 0.2))

    def training_step(self, batch):
        x, y = batch
        output = self.model(x)
        logits = output["logits"].squeeze(1).float()
        target = (y % 2).to(logits.dtype)

        loss_b = self.criterion(logits, target)
        loss_mse_rec = F.mse_loss(output["middle_freq_image"].float(), output["raw_reconstruction_delta"].float())

        fft_filter = self.model.fft_filter_module
        batch_size = x.shape[0]
        i_mask = fft_filter.i_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).detach()
        r_i_mask = fft_filter.r_i_mask.unsqueeze(0).expand(batch_size, -1, -1, -1).detach()
        all_mask = torch.ones_like(r_i_mask)

        loss_mse_mask_mid_frq = F.mse_loss(output["mask_mid_frq"].float(), i_mask.float())
        loss_mse_mask_mid_filtered = F.mse_loss(output["mask_mid_filtered"].float(), r_i_mask.float())
        loss_mse_mask_norm = F.mse_loss(
            output["mask_mid_frq"].float() + output["mask_mid_filtered"].float(),
            all_mask.float(),
        )
        loss_mse_mask = loss_mse_mask_mid_frq + loss_mse_mask_mid_filtered + loss_mse_mask_norm

        loss = self.bce_weight * loss_b + self.rec_weight * loss_mse_rec + self.mask_weight * loss_mse_mask
        self.log("train_loss", loss)
        self.log("train_loss_bce", loss_b)
        self.log("train_loss_rec", loss_mse_rec)
        self.log("train_loss_mask", loss_mse_mask)
        return loss
