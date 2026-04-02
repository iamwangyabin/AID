import os

import torch
import torch.nn as nn

from networks.vjepa2_1 import vision_transformer as vit
from utils.registry import MODELS


def _resolve_checkpoint_key(backbone_name, checkpoint_key):
    if checkpoint_key != "auto":
        return checkpoint_key
    if backbone_name in {"vit_base", "vit_large"}:
        return "ema_encoder"
    return "target_encoder"


def _clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _load_encoder_checkpoint(model, checkpoint_path, checkpoint_key):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    pretrained_dict = _clean_state_dict(checkpoint[checkpoint_key])

    current_state = model.state_dict()
    for key, value in current_state.items():
        if key not in pretrained_dict or pretrained_dict[key].shape != value.shape:
            pretrained_dict[key] = value

    msg = model.load_state_dict(pretrained_dict, strict=False)
    return msg


@MODELS.register_module(name="VJEPA2_1Linear")
class VJEPA2LinearProbe(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        backbone_name="vit_base",
        checkpoint_key="auto",
        image_size=384,
        num_frames=18,
        patch_size=16,
        tubelet_size=2,
        num_classes=1,
        freeze_backbone=True,
        pooling="mean",
        use_rope=True,
        uniform_power=True,
        use_sdpa=True,
        use_silu=False,
        wide_silu=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    ):
        super().__init__()
        self.checkpoint_path = os.path.expanduser(checkpoint_path)
        self.freeze_backbone = freeze_backbone
        self.num_frames = num_frames
        self.pooling = pooling

        if backbone_name not in vit.__dict__:
            raise ValueError(f"Unsupported V-JEPA 2.1 backbone: {backbone_name}")

        self.backbone = vit.__dict__[backbone_name](
            img_size=(image_size, image_size),
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            use_rope=use_rope,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
            use_silu=use_silu,
            wide_silu=wide_silu,
            img_temporal_dim_size=img_temporal_dim_size,
            interpolate_rope=interpolate_rope,
        )

        resolved_key = _resolve_checkpoint_key(backbone_name, checkpoint_key)
        msg = _load_encoder_checkpoint(self.backbone, self.checkpoint_path, resolved_key)
        print(f"Loaded V-JEPA 2.1 encoder from {self.checkpoint_path} with msg: {msg}")

        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _prepare_video_input(self, x):
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor of shape [B, C, H, W], got {tuple(x.shape)}")
        return x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)

    def _pool_tokens(self, tokens):
        if self.pooling == "mean":
            return tokens.mean(dim=1)
        if self.pooling == "max":
            return tokens.max(dim=1).values
        raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def forward(self, x):
        video = self._prepare_video_input(x)
        if self.freeze_backbone:
            with torch.no_grad():
                tokens = self.backbone(video)
        else:
            tokens = self.backbone(video)

        features = self._pool_tokens(tokens)
        logits = self.classifier(features)
        return {"logits": logits, "features": features}
