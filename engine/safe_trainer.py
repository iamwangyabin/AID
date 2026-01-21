import torch
import torch.nn as nn
import lightning as L
from torch.nn import functional as F

from utils.validate import validate
from utils.network_factory import get_model


class Trainer_SAFE(L.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.criterion = nn.CrossEntropyLoss()

    def training_step(self, batch):
        x, y = batch
        logits = self.model(x)["logits"]
        labels = (y % 2).long()
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model(x)["logits"]
        probs = F.softmax(logits, dim=1)[:, 1]
        self.validation_step_outputs_preds.append(probs)
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(
            torch.float32).flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(
            torch.float32).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        self.log("val_acc_epoch", acc, logger=True, sync_dist=True)
        self.log("val_ap_epoch", ap, logger=True, sync_dist=True)
        self.log("val_racc_epoch", r_acc, logger=True, sync_dist=True)
        self.log("val_facc_epoch", f_acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]
