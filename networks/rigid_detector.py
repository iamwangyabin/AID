import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoImageProcessor, AutoModel

from utils.registry import MODELS


@MODELS.register_module()
class RIGIDModel(nn.Module):
    """
    RIGID-style zero-shot detector:
    - add small Gaussian noise to image in pixel space
    - extract DINOv2 embeddings for original/noisy
    - score = 1 - cosine_similarity(emb_orig, emb_noisy)
    - output logits calibrated by (score - threshold)/temperature
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        noise_level: float = 0.02,
        threshold: float = 0.02,
        temperature: float = 0.01,
        num_noise_samples: int = 1,
        resize_shorter_edge: int | None = 256,
        crop_size: int | None = 224,
        interpolation: str = "bicubic",
    ):
        super().__init__()

        self.model_name = model_name
        self.noise_level = float(noise_level)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.num_noise_samples = int(num_noise_samples)
        self.resize_shorter_edge = resize_shorter_edge
        self.crop_size = crop_size
        self.interpolation = interpolation

        # Read DINOv2 input normalization stats from HF processor config,
        # but do NOT use processor() in forward (strategy2).
        processor = AutoImageProcessor.from_pretrained(model_name)
        mean = torch.tensor(processor.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(processor.image_std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _resize_shorter(self, x: torch.Tensor, shorter_edge: int) -> torch.Tensor:
        # x: [B,3,H,W]
        b, c, h, w = x.shape
        if min(h, w) == shorter_edge:
            return x
        scale = float(shorter_edge) / float(min(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return F.interpolate(x, size=(new_h, new_w), mode=self.interpolation, align_corners=False if self.interpolation in {"bilinear", "bicubic"} else None)

    def _center_crop(self, x: torch.Tensor, size: int) -> torch.Tensor:
        # x: [B,3,H,W]
        _, _, h, w = x.shape
        if h == size and w == size:
            return x
        top = max((h - size) // 2, 0)
        left = max((w - size) // 2, 0)
        return x[:, :, top : top + size, left : left + size]

    def _prep_pixel_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure x is float32 in [0,1] before noise & normalization.
        The recommended dataloader transform for this model is: Resize/CenterCrop + ToTensor (NO Normalize).
        """
        if x.dtype != torch.float32:
            x = x.float()

        # If looks like 0-255, rescale.
        if x.max() > 1.5:
            x = x / 255.0

        x = x.clamp(0.0, 1.0)

        if self.resize_shorter_edge is not None:
            x = self._resize_shorter(x, int(self.resize_shorter_edge))

        if self.crop_size is not None:
            x = self._center_crop(x, int(self.crop_size))
        return x

    def _normalize(self, x01: torch.Tensor) -> torch.Tensor:
        return (x01 - self._mean) / self._std

    def _get_embedding(self, x_norm: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x_norm)
        emb = out.last_hidden_state.mean(dim=1)  # [B, D]
        return F.normalize(emb, p=2, dim=1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B,3,H,W] tensor. Preferably after ToTensor only (no Normalize).

        Returns:
            dict with:
              - logits: [B,1] (higher => more likely fake)
              - rigid_score: [B,1]
              - stability: [B,1] cosine similarity (higher => more stable)
        """
        x01 = self._prep_pixel_tensor(x)

        # Compute fragility across K noise draws for stability.
        k = max(1, self.num_noise_samples)
        rigid_scores = []
        stabilities = []

        for _ in range(k):
            noise = torch.randn_like(x01) * self.noise_level
            x_noisy01 = (x01 + noise).clamp(0.0, 1.0)

            x_norm = self._normalize(x01)
            x_noisy_norm = self._normalize(x_noisy01)

            emb = self._get_embedding(x_norm)
            emb_noisy = self._get_embedding(x_noisy_norm)

            sim = F.cosine_similarity(emb, emb_noisy, dim=1)  # [B]
            rigid = 1.0 - sim  # [B]

            rigid_scores.append(rigid)
            stabilities.append(sim)

        rigid_score = torch.stack(rigid_scores, dim=0).mean(dim=0)  # [B]
        stability = torch.stack(stabilities, dim=0).mean(dim=0)  # [B]

        # Calibrated logit around threshold.
        temp = max(self.temperature, 1e-6)
        logits = (rigid_score - self.threshold) / temp

        return {
            "logits": logits.view(-1, 1),
            "rigid_score": rigid_score.view(-1, 1),
            "stability": stability.view(-1, 1),
        }