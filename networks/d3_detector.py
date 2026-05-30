import math
from contextlib import nullcontext

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODELS


CLIP_PENULTIMATE_DIMS = {
    "ViT-L/14": 1024,
}


class TransformerAttention(nn.Module):
    def __init__(self, input_dim, token_num, num_classes=1):
        super().__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc = nn.Linear(input_dim * token_num, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(k.size(-1))
        attention = self.softmax(attention)
        output = torch.matmul(attention, v)
        return self.fc(output.flatten(start_dim=1))


@MODELS.register_module(name="D3Model")
class D3Model(nn.Module):
    """D3 distorted feature discrepancy branch.

    Official CVPR 2025 D3 uses a frozen CLIP ViT-L/14 backbone, extracts the
    penultimate image feature from both patch-shuffled distorted inputs and the
    original image, then classifies the stacked feature sequence with an
    attention head.
    """

    def __init__(
        self,
        name="ViT-L/14",
        num_classes=1,
        shuffle_times=1,
        original_times=1,
        patch_size=14,
        freeze_backbone=True,
    ):
        super().__init__()
        if name not in CLIP_PENULTIMATE_DIMS:
            raise ValueError(f"D3Model currently supports {sorted(CLIP_PENULTIMATE_DIMS)}, got {name!r}.")

        self.name = name
        self.shuffle_times = int(shuffle_times)
        self.original_times = int(original_times)
        self.patch_size = int(patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size)
        self.freeze_backbone = freeze_backbone

        if self.shuffle_times < 0 or self.original_times < 1:
            raise ValueError("shuffle_times must be >= 0 and original_times must be >= 1.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be a positive integer.")

        self.model, self.preprocess = clip.load(name, device="cpu")
        self._penultimate_features = None
        self._register_penultimate_hook()

        feature_dim = CLIP_PENULTIMATE_DIMS[name]
        token_num = self.shuffle_times + self.original_times
        self.attention_head = TransformerAttention(feature_dim, token_num, num_classes)

        if self.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.model.eval()
        return self

    def _register_penultimate_hook(self):
        def hook(module, inputs, output):
            self._penultimate_features = output

        self.model.visual.ln_post.register_forward_hook(hook)

    def _shuffle_patches(self, x):
        bsz, channels, height, width = x.shape
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by patch_size={self.patch_size}."
            )

        patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        permutation = torch.randperm(patches.size(-1), device=x.device)
        shuffled = patches[:, :, permutation]
        return F.fold(
            shuffled,
            output_size=(height, width),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def _encode_penultimate(self, x):
        self._penultimate_features = None
        _ = self.model.encode_image(x)
        if self._penultimate_features is None:
            raise RuntimeError("Failed to capture CLIP penultimate visual features from ln_post.")
        return self._penultimate_features

    def _feature_context(self):
        return torch.no_grad() if self.freeze_backbone else nullcontext()

    def forward(self, x, return_feature=False):
        features = []

        with self._feature_context():
            for _ in range(self.shuffle_times):
                features.append(self._encode_penultimate(self._shuffle_patches(x)))

            original_features = self._encode_penultimate(x)
            for _ in range(self.original_times):
                features.append(original_features.clone())

        feature_tokens = torch.stack(features, dim=1)
        logits = self.attention_head(feature_tokens)

        output = {
            "logits": logits,
            "features": feature_tokens,
            "original_features": original_features,
        }
        if return_feature:
            output["penultimate_features"] = feature_tokens
        return output
