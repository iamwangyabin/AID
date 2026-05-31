import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from networks.cnndet import Bottleneck, ResNet, model_urls
from utils.registry import MODELS


@MODELS.register_module(name=["LOTAModel", "LOTA"])
class LOTAModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        self.backbone = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

        if pretrained:
            pretrained_dict = model_zoo.load_url(model_urls["resnet50"])
            pretrained_dict.pop("fc.weight", None)
            pretrained_dict.pop("fc.bias", None)
            self.backbone.load_state_dict(pretrained_dict, strict=False)

    def forward(self, x, return_feature=False, *args):
        if return_feature:
            logits, feature = self.backbone(x, return_feature=True)
            return {"logits": logits, "feature": feature}

        return self.backbone(x)
