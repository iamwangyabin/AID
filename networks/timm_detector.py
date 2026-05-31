import torch
import torch.nn as nn
import timm

from utils.registry import MODELS

@MODELS.register_module()
class TIMMModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(TIMMModel, self).__init__()
        self.backbone = timm.create_model(name, pretrained=True)
        self.backbone.reset_classifier(num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
            # if self.backbone.attn_pool is not None:
            #     features = self.backbone.attn_pool(self.backbone.forward_features(x))
            # elif self.backbone.global_pool == 'avg':
            #     features = self.backbone.forward_features(x)[:, self.backbone.num_prefix_tokens:].mean(dim=1)
            # elif self.backbone.global_pool:
            #     features = self.backbone.forward_features(x)[:, 0]  # class toke
        #     n
        return {'logits': self.backbone.forward_head(features),
                'features':features}
