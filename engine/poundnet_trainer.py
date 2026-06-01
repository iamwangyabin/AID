import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from engine.base_trainer import BaseTrainerModule
from utils.validate import validate
from utils.network_factory import get_model

mapping = {0: 0, 1: 20, 2: 1, 3: 21, 4: 2, 5: 22, 6: 3, 7: 23, 8: 4, 9: 24, 10: 5,
           11: 25, 12: 6, 13: 26, 14: 7, 15: 27, 16: 8, 17: 28, 18: 9, 19: 29, 20: 10,
           21: 30, 22: 11, 23: 31, 24: 12, 25: 32, 26: 13, 27: 33, 28: 14, 29: 34, 30: 15,
           31: 35, 32: 16, 33: 36, 34: 17, 35: 37, 36: 18, 37: 38, 38: 19, 39: 39}

def generate_mapping(base_number):
    n = base_number*2
    mapping = {}
    for k in range(n):
        if k % 2 == 0:
            mapping[k] = k // 2
        else:
            mapping[k] = base_number + (k - 1) // 2
    return mapping

class Trainer_PoundNet(BaseTrainerModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = get_model(opt)
        self.validation_step_outputs_gts, self.validation_step_outputs_preds = [], []
        self.celoss = nn.CrossEntropyLoss()

        self.mapping = generate_mapping(len(opt.datasets.train.multicalss_names))

    def training_step(self, batch):
        x, y = batch
        output = self.model(x, return_binary=True)
        logits = output['logits']
        b_logits = output['b_logits']

        loss = 0
        if self.opt.train.a != 0:
            # First, semantic alignment: classify y into 20 classes, then supervise the logits output to these 20 classes, regardless of real/fake.
            # Implement by splitting the logits dimension, grouping every 20, and performing cross entropy.
            cls_y = y//2 # 0~40 classes -> 0~19 classes. If prompt is 2, if it is 1, then it is...
            logits_groups = torch.chunk(logits, 2*self.opt.model.PROMPT_NUM_TEXT, dim=1)
            for i, logits_group in enumerate(logits_groups):
                loss += self.opt.train.a * F.cross_entropy(logits_group, cls_y)

        # Secondly, secondary semantic alignment: align the classification of fake samples with fake and real samples with real, also using cross entropy loss.
        # Implement this by chunking and masking, masking out the samples that belong to real, but it's useless...
        # for i, logits_group in enumerate(logits_groups):
        #     mask = i//2 == y % 2
        #     loss += 0.5*F.cross_entropy(logits_group[mask], cls_y[mask])

        # Perform deepfake detection within each subspace, i.e., perform deepfake detection at the class level.
        # 0 indicates real, 1 indicates fake, which is a deepfake detection task.
        # The main issue is that our y labels are 0, 1, 2, 3, 4, 5...38, 39 like this, where each class has two labels representing real and fake.
        # This results in 20 classes with 40 labels, meaning each class has both real and fake labels.
        # However, the logits are structured as 0-19 for real and then 20-39 for fake (when prompt=1), meaning it outputs the real for each class first, followed by the fake for each class.
        # add a mask to make this a 'real' binary cross-entropy loss, but seems no difference to final results
        # using mask is just like weighted cross-entropy, and we can't use real binary cross entropy for CLIP
        if self.opt.train.b != 0:
            new_y =  torch.tensor([self.mapping[label.item()] for label in y], dtype=torch.long, device=y.device)
            loss += self.opt.train.b * F.cross_entropy(logits, new_y)

        if self.opt.train.c != 0:
            # task alignment: unify all 20 classes into real/fake, and then unify all logits into real/fake (prompts mean).
            loss += self.opt.train.c * F.cross_entropy(b_logits, y % 2)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch):
        x, y = batch
        logits = self.model.forward_binary(x)['logits']
        self.validation_step_outputs_preds.append(F.softmax(logits, 1)[:,1])
        self.validation_step_outputs_gts.append(y)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.validation_step_outputs_preds, 0).to(torch.float32).flatten().cpu().numpy()
        all_gts = torch.cat(self.validation_step_outputs_gts, 0).to(torch.float32).cpu().numpy()
        acc, ap, r_acc, f_acc = validate(all_gts % 2, all_preds)
        self.log('val_acc_epoch', acc, logger=True, sync_dist=True)
        self.log('val_ap_epoch', ap, logger=True, sync_dist=True)
        self.log('val_racc_epoch', r_acc, logger=True, sync_dist=True)
        self.log('val_facc_epoch', f_acc, logger=True, sync_dist=True)
        self.validation_step_outputs_preds.clear()
        self.validation_step_outputs_gts.clear()

