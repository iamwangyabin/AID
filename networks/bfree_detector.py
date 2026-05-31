import math

import timm
import torch
import torch.nn as nn

from utils.registry import MODELS


def _tile_to_shape(x, target_hw):
    target_h, target_w = target_hw
    repeat_h = max(math.ceil(target_h / x.shape[-2]), 1)
    repeat_w = max(math.ceil(target_w / x.shape[-1]), 1)
    return x.repeat(1, 1, repeat_h, repeat_w)[..., :target_h, :target_w]


@MODELS.register_module()
class BFreeModel(nn.Module):
    """
    B-Free detector from CVPR 2025.

    This mirrors the official inference architecture: DINOv2 ViT with register
    tokens, a 504-pixel input grid, feature-space center/four-corner crops, and
    averaged crop logits.
    """

    def __init__(
        self,
        backbone_name="vit_base_patch14_reg4_dinov2.lvd142m",
        crop_size=504,
        num_classes=1,
        pretrained=True,
        freeze_backbone=False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.crop_size = int(crop_size)
        self.num_classes = int(num_classes)
        self.freeze_backbone = bool(freeze_backbone)

        model = timm.create_model(
            backbone_name,
            num_classes=self.num_classes,
            pretrained=pretrained,
        )
        model.set_input_size(img_size=self.crop_size)

        self.patch_embed = model.patch_embed
        model.patch_embed = nn.Identity()
        self.model = model

        if self.freeze_backbone:
            for name, param in self.model.named_parameters():
                if not name.startswith(("head.", "fc.")):
                    param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
            head = getattr(self.model, "head", None)
            if head is not None:
                head.train(mode)
        return self

    def _crop_embeddings(self, embeddings):
        target_h, target_w = (int(v) for v in self.patch_embed.grid_size)
        if embeddings.shape[-2] < target_h or embeddings.shape[-1] < target_w:
            embeddings = _tile_to_shape(embeddings, (target_h, target_w))

        max_h = max(embeddings.shape[-2] - target_h, 0)
        max_w = max(embeddings.shape[-1] - target_w, 0)
        center_h = max_h // 2
        center_w = max_w // 2

        crops = (
            embeddings[..., center_h : center_h + target_h, center_w : center_w + target_w],
            embeddings[..., :target_h, :target_w],
            embeddings[..., -target_h:, :target_w],
            embeddings[..., -target_h:, -target_w:],
            embeddings[..., :target_h, -target_w:],
        )
        return torch.cat(crops, dim=0)

    def _forward_logits(self, x):
        embeddings = self.patch_embed.proj(x)
        embeddings = self._crop_embeddings(embeddings)
        if self.patch_embed.flatten:
            embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.patch_embed.norm(embeddings)

        logits = self.model(embeddings)
        return torch.stack(torch.split(logits, x.shape[0], dim=0), dim=0).mean(dim=0)

    def forward(self, x):
        return {"logits": self._forward_logits(x)}
