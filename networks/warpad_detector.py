import inspect
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_wavelets import DWTForward, DWTInverse

from utils.registry import MODELS


@MODELS.register_module()
class WaRPADModel(nn.Module):
    """
    WaRPAD (training-free) detector adapted to this repo's evaluation pipeline.

    Core idea (per-image score):
      1) resize to `prep_size` and ImageNet-normalize
      2) split image into non-overlapping `patch_size` patches
      3) wavelet-based high-frequency perturbation per patch
      4) DINOv2 features for original vs perturbed patches (cls token)
      5) similarity = cosine(feat, feat_pert)
      6) per-image similarity_mean = mean(similarity over patches)
      7) warpad_score = 1 - similarity_mean  (higher => more likely fake)
      8) logits = (warpad_score - threshold) / temperature

    Returns a dict with:
      - logits: [B,1] (higher => more likely fake)
      - warpad_score: [B,1]
      - similarity_mean: [B,1]
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dino_model: str = "dinov2_vitl14",
        prep_size: int = 896,
        patch_size: int = 224,
        noise_level: float = 0.1,
        threshold: float = 0.02,
        temperature: float = 0.01,
        wavelet_J: int = 2,
        wave: str = "haar",
        interpolation: str = "bicubic",
    ) -> None:
        super().__init__()

        self.dino_model = str(dino_model)
        self.prep_size = int(prep_size)
        self.patch_size = int(patch_size)
        self.noise_level = float(noise_level)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.wavelet_J = int(wavelet_J)
        self.wave = str(wave)
        self.interpolation = str(interpolation)

        if self.prep_size <= 0 or self.patch_size <= 0:
            raise ValueError("prep_size and patch_size must be positive.")
        if (self.prep_size % self.patch_size) != 0:
            raise ValueError(
                f"prep_size must be divisible by patch_size. Got prep_size={self.prep_size}, patch_size={self.patch_size}."
            )

        # DINOv2 from torch.hub (kept as in your provided code).
        # NOTE: will download weights on first run.
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.dino_model)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Wavelet operators (train-free, used in forward)
        self.dwt = DWTForward(J=self.wavelet_J, wave=self.wave)
        self.idwt = DWTInverse(wave=self.wave)

        mean = torch.tensor(self.IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

    def _prep_pixel_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure x is float32 in [0,1], resize to [prep_size, prep_size] if needed.

        Recommended dataloader transform for this model:
          Resize(prep_size) + CenterCrop(prep_size) + ToTensor()
        (NO Normalize)
        """
        if x.dtype != torch.float32:
            x = x.float()

        # If looks like 0-255, rescale.
        if x.max() > 1.5:
            x = x / 255.0

        x = x.clamp(0.0, 1.0)

        b, c, h, w = x.shape
        if c != 3:
            raise ValueError(f"Expected 3-channel images, got C={c}.")

        if h != self.prep_size or w != self.prep_size:
            x = F.interpolate(
                x,
                size=(self.prep_size, self.prep_size),
                mode=self.interpolation,
                align_corners=False if self.interpolation in {"bilinear", "bicubic"} else None,
            )
        return x

    def _normalize(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / self._std

    def _forward_features_clstoken(self, x: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 torch.hub models expose forward_features and use key 'x_norm_clstoken'.
        But the signature differs by version; support both:
          forward_features(x)
          forward_features(x, masks)
        """
        ff = getattr(self.backbone, "forward_features", None)
        if ff is None:
            raise RuntimeError("Loaded DINOv2 model has no forward_features().")

        try:
            out = ff(x, None)
        except TypeError:
            out = ff(x)

        if not isinstance(out, dict) or "x_norm_clstoken" not in out:
            raise RuntimeError("Unexpected forward_features output; missing key 'x_norm_clstoken'.")
        return out["x_norm_clstoken"]

    def _split_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        x: [B,3,H,W] where H=W=prep_size and divisible by patch_size
        returns:
          patches: [B*num_patches, 3, patch_size, patch_size]
          num_patches: int per image
        """
        b, c, h, w = x.shape
        p = self.patch_size
        if (h % p) != 0 or (w % p) != 0:
            raise ValueError(
                f"Input HxW must be divisible by patch_size. Got {h}x{w}, patch_size={p}."
            )

        patches = x.unfold(2, p, p).unfold(3, p, p)  # [B,C,Hp,Wp,p,p]
        patches = patches.reshape([b, c, -1, p, p]).transpose(1, 2)  # [B,num_patches,C,p,p]
        num_patches = patches.shape[1]
        patches = patches.reshape([-1, c, p, p])  # [B*num_patches,C,p,p]
        return patches, num_patches

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
          x: [B,3,H,W], preferably after ToTensor only (no Normalize).

        Returns:
          dict with logits compatible with [`utils.validate_plain()`](utils/validate.py:61).
        """
        x01 = self._prep_pixel_tensor(x)
        x_norm = self._normalize(x01)

        patches, num_patches = self._split_patches(x_norm)
        yl, yh = self.dwt(patches)
        yl_zeros = torch.zeros_like(yl)
        pert_hf = self.idwt((yl_zeros, yh))
        perturbed = patches - self.noise_level * pert_hf

        feats = self._forward_features_clstoken(patches)
        feats_pert = self._forward_features_clstoken(perturbed)

        similarity = F.cosine_similarity(feats, feats_pert, dim=-1)  # [B*num_patches]
        similarity = similarity.view(-1, num_patches)  # [B, num_patches]
        similarity_mean = similarity.mean(dim=1)  # [B]

        warpad_score = 1.0 - similarity_mean  # [B]
        temp = max(self.temperature, 1e-6)
        logits = (warpad_score - self.threshold) / temp

        return {
            "logits": logits.view(-1, 1),
            "warpad_score": warpad_score.view(-1, 1),
            "similarity_mean": similarity_mean.view(-1, 1),
        }