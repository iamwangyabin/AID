import torch
import torch.nn as nn

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


    def configure_optimizers(self):
        optparams = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = self.opt.train.optimizer(optparams)
        scheduler = self.opt.train.scheduler(optimizer)
        return [optimizer], [scheduler]

