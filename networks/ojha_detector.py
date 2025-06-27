import clip
import os
import torch
import torch.nn as nn

from utils.registry import MODELS

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "RN50x64": 1024,
    "ViT-L/14@336px": 768,
}

@MODELS.register_module()
class OjhaModel(nn.Module):
    def __init__(self, name="ViT-L/14", num_classes=1):
        super(OjhaModel, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        for param in self.model.parameters():
            param.requires_grad = False
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x, **kwargs):
        features = self.model.encode_image(x)

        return {'logits': self.fc(features),
                'features': features}

