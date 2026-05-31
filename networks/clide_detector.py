from __future__ import annotations

# Adapted from the official CLIDE implementation:
# https://github.com/FujitsuResearch/domain-adaptive-image-detection
# Original repository license: CC BY-NC 4.0.

from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import clip
except ImportError:
    from networks.SPrompts.clip import clip

from utils.registry import MODELS


def _torch_load(path: str | Path):
    """Load torch files across PyTorch versions that differ on weights_only."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def sphx(x: torch.Tensor, m: int = 500, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PCA whitening / sphering from the official CLIDE implementation.

    Args:
        x: [N, D] embedding tensor.
        m: number of principal components to keep.
        eps: numerical floor for eigenvalues.

    Returns:
        sph: [N, m] whitened embeddings.
        rotation_matrix: [D, m] whitening matrix.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got {x.ndim}D.")
    n, d = x.shape
    if n < 2:
        raise ValueError("At least two embeddings are required for whitening.")

    m = min(int(m), d, max(1, n - 1))
    x = x.float()
    x_mean = x.mean(dim=0)
    xu = x - x_mean

    cov_matrix = torch.cov(xu.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    indices = torch.argsort(eigenvalues, descending=True)[:m]

    s_inv_sqrt = torch.diag(torch.rsqrt(eigenvalues[indices].clamp_min(eps)))
    v_matrix = eigenvectors[:, indices]
    rotation_matrix = v_matrix @ s_inv_sqrt
    sph = xu @ rotation_matrix
    return sph, rotation_matrix


@MODELS.register_module(name=["CLIDEModel", "CLIDE"])
class CLIDEModel(nn.Module):
    """
    CLIDE: conditional-likelihood zero-shot generated-image detector.

    This follows the official FujitsuResearch/domain-adaptive-image-detection
    implementation:
      - OpenAI CLIP ViT-L/14 image embeddings
      - global whitening from a precomputed (w_mat, w_mean), or
      - local whitening from top-k representative embeddings
      - Gaussian log-likelihood in the whitened CLIP space

    AID's evaluation expects larger logits to mean "more likely fake". Natural
    images usually have higher CLIDE likelihood under a real-image representative
    distribution, so the default score is -likelihood.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        name: str = "ViT-L/14",
        mode: Literal["global", "local"] = "global",
        w_mat_path: str | None = None,
        rep_mat_path: str | None = None,
        k: int = 500,
        m: int = 400,
        threshold: float = 0.0,
        temperature: float = 1000.0,
        score_sign: float = -1.0,
        image_size: int = 224,
        resize_shorter_edge: int | None = 224,
        interpolation: str = "bicubic",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if mode not in {"global", "local"}:
            raise ValueError(f"mode must be 'global' or 'local', got {mode!r}.")

        self.name = str(name)
        self.mode = mode
        self.k = int(k)
        self.m = int(m)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.score_sign = float(score_sign)
        self.image_size = int(image_size)
        self.resize_shorter_edge = resize_shorter_edge
        self.interpolation = str(interpolation)
        self.eps = float(eps)

        self.clip_model, _ = clip.load(self.name, device="cpu")
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        mean = torch.tensor(self.CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_mean", mean, persistent=False)
        self.register_buffer("_std", std, persistent=False)

        if self.mode == "global":
            if not w_mat_path:
                raise ValueError("CLIDE global mode requires model.w_mat_path.")
            w_mat, w_mean = self._load_global_stats(w_mat_path)
            self.register_buffer("_w_mat", w_mat.float(), persistent=False)
            self.register_buffer("_w_mean", w_mean.float(), persistent=False)
            self.register_buffer("_rep_mat", torch.empty(0), persistent=False)
        else:
            if not rep_mat_path:
                raise ValueError("CLIDE local mode requires model.rep_mat_path.")
            rep_mat = self._load_rep_stats(rep_mat_path)
            self.register_buffer("_rep_mat", rep_mat.float(), persistent=False)
            self.register_buffer("_w_mat", torch.empty(0), persistent=False)
            self.register_buffer("_w_mean", torch.empty(0), persistent=False)

    @staticmethod
    def _load_global_stats(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"CLIDE whitening stats not found: {path}. "
                "Download the official file with tools/download_clide_official_stats.py."
            )
        obj = _torch_load(path)
        if not isinstance(obj, (tuple, list)) or len(obj) != 2:
            raise ValueError(f"Expected (w_mat, w_mean) in {path}.")
        w_mat, w_mean = obj
        if w_mat.ndim != 2 or w_mean.ndim != 1:
            raise ValueError(f"Invalid CLIDE whitening tensors in {path}.")
        return w_mat, w_mean

    @staticmethod
    def _load_rep_stats(path: str | Path) -> torch.Tensor:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"CLIDE representative matrix not found: {path}. "
                "Download the official file with tools/download_clide_official_stats.py."
            )
        rep_mat = _torch_load(path)
        if not torch.is_tensor(rep_mat) or rep_mat.ndim != 2:
            raise ValueError(f"Expected a [N, D] tensor in {path}.")
        return rep_mat

    def _resize_shorter(self, x: torch.Tensor, shorter_edge: int) -> torch.Tensor:
        _, _, h, w = x.shape
        if min(h, w) == shorter_edge:
            return x
        scale = float(shorter_edge) / float(min(h, w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return F.interpolate(
            x,
            size=(new_h, new_w),
            mode=self.interpolation,
            align_corners=False if self.interpolation in {"bilinear", "bicubic"} else None,
        )

    def _center_crop(self, x: torch.Tensor, size: int) -> torch.Tensor:
        _, _, h, w = x.shape
        if h == size and w == size:
            return x
        top = max((h - size) // 2, 0)
        left = max((w - size) // 2, 0)
        return x[:, :, top : top + size, left : left + size]

    def _prep_pixel_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        x = x.clamp(0.0, 1.0)

        if self.resize_shorter_edge is not None:
            x = self._resize_shorter(x, int(self.resize_shorter_edge))
        if self.image_size is not None:
            x = self._center_crop(x, int(self.image_size))

        return (x - self._mean) / self._std

    @torch.no_grad()
    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # The official code uses raw CLIP image embeddings, not L2-normalized
        # embeddings. Cosine normalization is used only for local neighbor search.
        emb = self.clip_model.encode_image(x)
        return emb.float()

    def _global_likelihood(self, embeddings: torch.Tensor) -> torch.Tensor:
        w_mat = self._w_mat.to(device=embeddings.device, dtype=embeddings.dtype)
        w_mean = self._w_mean.to(device=embeddings.device, dtype=embeddings.dtype)
        m = w_mat.shape[1]
        log_const = 0.5 * m * torch.log(torch.tensor(2 * np.pi, device=embeddings.device, dtype=embeddings.dtype))
        whitened = (embeddings - w_mean) @ w_mat
        return -(log_const + 0.5 * whitened.square().sum(dim=1))

    def _local_likelihood(self, embeddings: torch.Tensor) -> torch.Tensor:
        rep_mat = self._rep_mat.to(device=embeddings.device, dtype=embeddings.dtype)
        if rep_mat.shape[0] < 2:
            raise ValueError("CLIDE local mode requires at least two representative embeddings.")

        k = min(max(2, self.k), rep_mat.shape[0])
        m = min(self.m, k - 1, rep_mat.shape[1])
        log_const = 0.5 * m * torch.log(torch.tensor(2 * np.pi, device=embeddings.device, dtype=embeddings.dtype))

        rep_norm = F.normalize(rep_mat, p=2, dim=1)
        emb_norm = F.normalize(embeddings, p=2, dim=1)

        likelihoods = []
        for emb, emb_n in zip(embeddings, emb_norm):
            similarities = emb_n @ rep_norm.T
            top_k_indices = torch.topk(similarities, k=k, largest=True).indices
            selected_rep = rep_mat[top_k_indices]

            _, local_w = sphx(selected_rep, m=m, eps=self.eps)
            whitened_embedding = (emb - selected_rep.mean(dim=0)) @ local_w
            likelihood = -(log_const + 0.5 * whitened_embedding.square().sum())
            likelihoods.append(likelihood)

        return torch.stack(likelihoods, dim=0)

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        x = self._prep_pixel_tensor(x)
        embeddings = self._encode_image(x)

        if self.mode == "global":
            likelihood = self._global_likelihood(embeddings)
        else:
            likelihood = self._local_likelihood(embeddings)

        clide_score = likelihood * self.score_sign
        temp = max(self.temperature, 1e-6)
        logits = (clide_score - self.threshold) / temp

        return {
            "logits": logits.view(-1, 1),
            "clide_score": clide_score.view(-1, 1),
            "likelihood": likelihood.view(-1, 1),
            "features": embeddings,
        }
