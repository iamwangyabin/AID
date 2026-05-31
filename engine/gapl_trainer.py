from engine.base_trainer import Trainer


class Trainer_GAPL(Trainer):
    def _split_batch(self, batch):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("GAPL trainer expects a batch shaped as (image, label[, generator]).")

    def training_step(self, batch):
        x, y = self._split_batch(batch)
        logits = self.model(x)["logits"]
        loss = self.criterion(logits.squeeze(1), (y % 2).to(self.dtype))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = self._split_batch(batch)
        logits = self.model(x)["logits"]
        self.validation_step_outputs_preds.append(logits.squeeze(1))
        self.validation_step_outputs_gts.append(y)
