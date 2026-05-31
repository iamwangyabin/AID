from __future__ import annotations

from collections.abc import Sequence as SequenceABC
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.transforms import functional as TVF

from utils.registry import MODELS


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def _as_pair(value) -> tuple[int, int]:
    if isinstance(value, SequenceABC):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def generate_circular_mask(
    height: int,
    width: int,
    radius_start: int,
    radius_stop: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    y = torch.arange(height, device=device, dtype=dtype) - height // 2
    x = torch.arange(width, device=device, dtype=dtype) - width // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius = torch.sqrt(xx.square() + yy.square())
    mask = (radius < float(radius_start)).to(dtype)
    if radius_stop is not None:
        mask = torch.where(radius > float(radius_stop), torch.ones_like(mask), mask)
    return mask.view(1, 1, height, width)


def filter_image_frequencies(
    image: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    spectrum = torch.fft.fftshift(torch.fft.fft2(image.float()), dim=(-2, -1))
    low_freq = spectrum * mask
    high_freq = spectrum * (1.0 - mask)
    low_freq = torch.fft.ifft2(torch.fft.ifftshift(low_freq, dim=(-2, -1))).real
    high_freq = torch.fft.ifft2(torch.fft.ifftshift(high_freq, dim=(-2, -1))).real
    return low_freq, high_freq


class FrequencyLoss(nn.Module):
    """Patch-based FFT loss used by SPAI's masked frequency modeling stage."""

    def __init__(
        self,
        loss_gamma: float = 1.0,
        matrix_gamma: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        with_matrix: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
    ) -> None:
        super().__init__()
        self.loss_gamma = loss_gamma
        self.matrix_gamma = matrix_gamma
        self.patch_factor = patch_factor
        self.ave_spectrum = ave_spectrum
        self.with_matrix = with_matrix
        self.log_matrix = log_matrix
        self.batch_matrix = batch_matrix

    def tensor2freq(self, x: torch.Tensor) -> torch.Tensor:
        patch_factor = int(self.patch_factor)
        _, _, h, w = x.shape
        if h % patch_factor != 0 or w % patch_factor != 0:
            raise ValueError("patch_factor must divide image height and width.")

        patch_h = h // patch_factor
        patch_w = w // patch_factor
        patches = []
        for i in range(patch_factor):
            for j in range(patch_factor):
                patches.append(x[:, :, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])

        y = torch.stack(patches, dim=1).float()
        freq = torch.fft.fftshift(torch.fft.fft2(y, norm="ortho"), dim=(-2, -1))
        return torch.stack([freq.real, freq.imag], dim=-1)

    def loss_formulation(
        self,
        recon_freq: torch.Tensor,
        real_freq: torch.Tensor,
        matrix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        diff = (recon_freq - real_freq).square()
        loss = torch.sqrt(diff[..., 0] + diff[..., 1] + 1e-12).pow(self.loss_gamma)

        if not self.with_matrix:
            return loss

        if matrix is not None:
            weight_matrix = matrix.detach()
        else:
            matrix_tmp = torch.sqrt(diff[..., 0] + diff[..., 1]).pow(self.matrix_gamma)
            if self.log_matrix:
                matrix_tmp = torch.log(matrix_tmp + 1.0)
            if self.batch_matrix:
                matrix_tmp = matrix_tmp / matrix_tmp.max().clamp_min(1e-12)
            else:
                denom = matrix_tmp.max(-1).values.max(-1).values[:, :, :, None, None].clamp_min(1e-12)
                matrix_tmp = matrix_tmp / denom
            matrix_tmp = torch.nan_to_num(matrix_tmp, nan=0.0).clamp(0.0, 1.0)
            weight_matrix = matrix_tmp.detach()

        return weight_matrix * loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor, matrix: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred_freq = self.tensor2freq(pred)
        target_freq = self.tensor2freq(target)
        if self.ave_spectrum:
            pred_freq = torch.mean(pred_freq, dim=0, keepdim=True)
            target_freq = torch.mean(target_freq, dim=0, keepdim=True)
        return self.loss_formulation(pred_freq, target_freq, matrix)


class Projector(nn.Module):
    def __init__(
        self,
        proj_layers: int,
        input_dim: int,
        proj_dim: int,
        last_layer_activation=nn.GELU,
        input_norm: bool = True,
        output_norm: bool = True,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(input_dim) if input_norm else nn.Identity()
        layers: list[nn.Module] = [nn.Dropout(dropout)]
        for i in range(proj_layers):
            layers.extend(
                [
                    nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                    nn.GELU() if i < proj_layers - 1 else last_layer_activation(),
                    nn.Dropout(dropout),
                ]
            )
        self.projector = nn.Sequential(*layers)
        self.norm2 = nn.LayerNorm(proj_dim) if output_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm2(self.projector(self.norm1(x)))


class FeatureSpecificProjector(nn.Module):
    def __init__(
        self,
        intermediate_features_num: int,
        proj_layers: int,
        input_dim: int,
        proj_dim: int,
        last_layer_activation=nn.GELU,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                Projector(proj_layers, input_dim, proj_dim, last_layer_activation, dropout=dropout)
                for _ in range(intermediate_features_num)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = [projector(x[:, i]) for i, projector in enumerate(self.projectors)]
        return torch.stack(projected, dim=1)


class FeatureImportanceProjector(nn.Module):
    def __init__(
        self,
        intermediate_features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.randn([1, intermediate_features_num, proj_dim]))
        self.proj1 = Projector(proj_layers, 2 * input_dim, proj_dim, input_norm=False, dropout=dropout)
        self.proj2 = Projector(proj_layers, proj_dim, proj_dim, input_norm=False, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=2)
        x_std = x.std(dim=2)
        x = torch.cat([x_mean, x_std], dim=-1)
        x = self.proj1(x)
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)
        return self.proj2(x)


class FrequencyRestorationEstimator(nn.Module):
    def __init__(
        self,
        features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        patch_projection: bool = True,
        patch_projection_per_feature: bool = True,
        proj_last_layer_activation_type: Optional[str] = None,
        original_image_features_branch: bool = True,
        dropout: float = 0.5,
        disable_reconstruction_similarity: bool = False,
    ) -> None:
        super().__init__()
        self.features_num = int(features_num)
        self.proj_dim = int(proj_dim)

        if proj_last_layer_activation_type == "gelu":
            last_activation = nn.GELU
        elif proj_last_layer_activation_type is None:
            last_activation = nn.Identity
        else:
            raise ValueError(f"Unsupported projector activation: {proj_last_layer_activation_type}")

        if patch_projection and patch_projection_per_feature:
            self.patch_projector = FeatureSpecificProjector(
                features_num, proj_layers, input_dim, proj_dim, last_activation, dropout=dropout
            )
        elif patch_projection:
            self.patch_projector = Projector(proj_layers, input_dim, proj_dim, last_activation, dropout=dropout)
        else:
            self.patch_projector = nn.Identity()

        self.original_features_processor = None
        if original_image_features_branch:
            self.original_features_processor = FeatureImportanceProjector(
                features_num, proj_dim, proj_dim, proj_layers, dropout=dropout
            )

        self.disable_reconstruction_similarity = disable_reconstruction_similarity
        if self.disable_reconstruction_similarity and self.original_features_processor is None:
            raise ValueError("disable_reconstruction_similarity requires original_image_features_branch=True.")

    @property
    def output_dim(self) -> int:
        if self.disable_reconstruction_similarity:
            return self.proj_dim
        sim_dim = 6 * self.features_num
        if self.original_features_processor is None:
            return sim_dim
        return sim_dim + self.proj_dim

    def forward(self, x: torch.Tensor, low_freq: torch.Tensor, high_freq: torch.Tensor) -> torch.Tensor:
        orig = self.patch_projector(x)
        low_freq = self.patch_projector(low_freq)
        high_freq = self.patch_projector(high_freq)

        if self.disable_reconstruction_similarity:
            return self.original_features_processor(orig)

        sim_x_low = F.cosine_similarity(orig, low_freq, dim=-1)
        sim_x_high = F.cosine_similarity(orig, high_freq, dim=-1)
        sim_low_high = F.cosine_similarity(low_freq, high_freq, dim=-1)

        features = torch.cat(
            [
                sim_x_low.mean(dim=-1),
                sim_x_low.std(dim=-1),
                sim_x_high.mean(dim=-1),
                sim_x_high.std(dim=-1),
                sim_low_high.mean(dim=-1),
                sim_low_high.std(dim=-1),
            ],
            dim=1,
        )

        if self.original_features_processor is not None:
            features = torch.cat([features, self.original_features_processor(orig)], dim=1)
        return features


class ClassificationHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 1, mlp_ratio: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        hidden = input_dim * mlp_ratio
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class SpectralContextAttention(nn.Module):
    def __init__(self, input_dim: int, attn_embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if attn_embed_dim % num_heads != 0:
            raise ValueError("attn_embed_dim must be divisible by num_heads.")
        self.heads = num_heads
        self.dim_head = attn_embed_dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.to_kv = nn.Linear(input_dim, attn_embed_dim * 2, bias=False)
        self.patch_aggregator = nn.Parameter(torch.zeros(num_heads, 1, self.dim_head))
        nn.init.trunc_normal_(self.patch_aggregator, std=0.02)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Sequential(nn.Linear(attn_embed_dim, input_dim, bias=False), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        aggregator = self.patch_aggregator.expand(batch, -1, -1, -1)
        k, v = self.to_kv(x).chunk(2, dim=-1)
        k = k.view(batch, x.size(1), self.heads, self.dim_head).permute(0, 2, 1, 3)
        v = v.view(batch, x.size(1), self.heads, self.dim_head).permute(0, 2, 1, 3)
        attn = self.attend(torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(batch, 1, self.heads * self.dim_head)
        return self.to_out(x).squeeze(1)


class TimmIntermediateViT(nn.Module):
    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_layers: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        try:
            self.backbone = timm.create_model(
                backbone_name, pretrained=pretrained, num_classes=0, dynamic_img_size=True
            )
        except TypeError:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)

        if not hasattr(self.backbone, "blocks"):
            raise ValueError(f"SPAI requires a timm ViT-like backbone with blocks, got {backbone_name}.")

        blocks = list(self.backbone.blocks)
        if feature_layers is None:
            feature_layers = list(range(len(blocks)))
        self.feature_layers = [idx if idx >= 0 else len(blocks) + idx for idx in feature_layers]
        for idx in self.feature_layers:
            if idx < 0 or idx >= len(blocks):
                raise ValueError(f"feature layer {idx} is out of range for {backbone_name}.")

        self.embed_dim = int(getattr(self.backbone, "num_features"))
        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.patch_size = _as_pair(getattr(self.backbone.patch_embed, "patch_size", (16, 16)))[0]
        self._features: list[torch.Tensor] = []
        self._hooks = []
        selected = set(self.feature_layers)
        for idx, block in enumerate(blocks):
            if idx in selected:
                self._hooks.append(block.register_forward_hook(self._make_hook()))

    def _make_hook(self):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            self._features.append(output)
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features = []
        self.backbone.forward_features(x)
        if len(self._features) != len(self.feature_layers):
            raise RuntimeError(
                f"Expected {len(self.feature_layers)} intermediate features, got {len(self._features)}."
            )
        tokens = []
        for feat in self._features:
            if feat.ndim != 3:
                raise RuntimeError(f"Expected ViT token features [B, L, D], got {tuple(feat.shape)}.")
            tokens.append(feat[:, self.num_prefix_tokens:, :])
        return torch.stack(tokens, dim=1)


class SPAIBase(nn.Module):
    def __init__(
        self,
        input_is_normalized: bool = False,
        input_mean: Sequence[float] = IMAGENET_MEAN,
        input_std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        self.input_is_normalized = input_is_normalized
        self.register_buffer("_input_mean", torch.tensor(input_mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_input_std", torch.tensor(input_std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_mean", torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_std", torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()
        if self.input_is_normalized:
            x = x * self._input_std + self._input_mean
        elif x.max() > 1.5:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / self._std


@MODELS.register_module(name="SPAIModel")
class SPAIModel(SPAIBase):
    """
    SPAI-style detector with masked spectral reconstruction similarity and
    spectral context attention for arbitrary-resolution images.
    """

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        mfm_checkpoint: Optional[str] = None,
        feature_layers: Optional[Sequence[int]] = None,
        projection_dim: int = 1024,
        projection_layers: int = 2,
        patch_projection: bool = True,
        patch_projection_per_feature: bool = True,
        projector_last_layer_activation_type: Optional[str] = None,
        original_image_features_branch: bool = True,
        disable_reconstruction_similarity: bool = False,
        masking_radius: int = 16,
        resolution_mode: str = "arbitrary",
        img_size: int = 224,
        patch_stride: int = 224,
        minimum_patches: int = 4,
        feature_extraction_batch_size: int = 64,
        attn_embed_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 1,
        cls_mlp_ratio: int = 3,
        dropout: float = 0.5,
        freeze_backbone: bool = True,
        input_is_normalized: bool = False,
        input_mean: Sequence[float] = IMAGENET_MEAN,
        input_std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__(input_is_normalized=input_is_normalized, input_mean=input_mean, input_std=input_std)
        if resolution_mode not in {"fixed", "arbitrary"}:
            raise ValueError("resolution_mode must be 'fixed' or 'arbitrary'.")

        self.resolution_mode = resolution_mode
        self.img_size = int(img_size)
        self.patch_stride = int(patch_stride)
        self.minimum_patches = int(minimum_patches)
        self.feature_extraction_batch_size = int(feature_extraction_batch_size)
        self.masking_radius = int(masking_radius)
        self.freeze_backbone = bool(freeze_backbone)

        self.backbone = TimmIntermediateViT(backbone_name, pretrained=pretrained, feature_layers=feature_layers)
        feature_count = len(self.backbone.feature_layers)
        self.features_processor = FrequencyRestorationEstimator(
            features_num=feature_count,
            input_dim=self.backbone.embed_dim,
            proj_dim=projection_dim,
            proj_layers=projection_layers,
            patch_projection=patch_projection,
            patch_projection_per_feature=patch_projection_per_feature,
            proj_last_layer_activation_type=projector_last_layer_activation_type,
            original_image_features_branch=original_image_features_branch,
            dropout=dropout,
            disable_reconstruction_similarity=disable_reconstruction_similarity,
        )

        cls_vector_dim = 6 * feature_count
        if original_image_features_branch and disable_reconstruction_similarity:
            cls_vector_dim = projection_dim
        elif original_image_features_branch:
            cls_vector_dim += projection_dim
        self.cls_vector_dim = cls_vector_dim

        self.patch_attention = SpectralContextAttention(
            cls_vector_dim, attn_embed_dim=attn_embed_dim, num_heads=num_heads, dropout=dropout
        )
        self.patch_norm = nn.LayerNorm(cls_vector_dim)
        out_classes = num_classes if num_classes > 2 else 1
        self.cls_head = ClassificationHead(cls_vector_dim, out_classes, mlp_ratio=cls_mlp_ratio, dropout=dropout)
        self.features_processor.apply(_init_weights)
        self.patch_attention.apply(_init_weights)
        self.patch_norm.apply(_init_weights)
        self.cls_head.apply(_init_weights)

        if mfm_checkpoint:
            self.load_mfm_backbone(mfm_checkpoint)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def load_mfm_backbone(self, checkpoint_path: str) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state = {}
        for key, value in state_dict.items():
            for prefix in ("model.backbone.", "backbone."):
                if key.startswith(prefix):
                    backbone_state[key[len(prefix):]] = value
                    break
        if not backbone_state:
            print(f"No SPAI MFM backbone weights found in {checkpoint_path}; skipping.")
            return
        msg = self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded SPAI MFM backbone from {checkpoint_path}: {msg}")

    def _frequency_views(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x01.shape
        mask = generate_circular_mask(h, w, self.masking_radius, device=x01.device, dtype=torch.float32)
        low_freq, high_freq = filter_image_frequencies(x01, mask)
        return low_freq.clamp(0.0, 1.0).to(x01.dtype), high_freq.clamp(0.0, 1.0).to(x01.dtype)

    def _fre_vector(self, x01: torch.Tensor) -> torch.Tensor:
        low_freq, high_freq = self._frequency_views(x01)
        x = self._normalize(x01)
        low_freq = self._normalize(low_freq)
        high_freq = self._normalize(high_freq)
        if self.freeze_backbone:
            with torch.no_grad():
                x_feat = self.backbone(x)
                low_feat = self.backbone(low_freq)
                high_feat = self.backbone(high_freq)
        else:
            x_feat = self.backbone(x)
            low_feat = self.backbone(low_freq)
            high_feat = self.backbone(high_freq)
        return self.features_processor(x_feat, low_feat, high_feat)

    def _resize_to_patch(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[-2:] == (self.img_size, self.img_size):
            return x01
        return F.interpolate(x01, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)

    def _patchify_batch(self, x01: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x01.shape
        patch = self.img_size
        if h < patch or w < patch:
            x01 = F.interpolate(x01, size=(max(h, patch), max(w, patch)), mode="bicubic", align_corners=False)
            b, c, h, w = x01.shape

        patches = x01.unfold(2, patch, self.patch_stride).unfold(3, patch, self.patch_stride)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous().view(b, -1, c, patch, patch)

        if patches.size(1) >= self.minimum_patches:
            return patches

        fallback = []
        for img in x01:
            if img.shape[-2] < patch or img.shape[-1] < patch:
                img = F.interpolate(img.unsqueeze(0), size=(patch, patch), mode="bicubic", align_corners=False).squeeze(0)
            crops = TVF.five_crop(img, [patch, patch])
            fallback.append(torch.stack(list(crops), dim=0))
        return torch.stack(fallback, dim=0)

    def _vectors_in_chunks(self, patches: torch.Tensor) -> torch.Tensor:
        vectors = []
        chunk = max(1, self.feature_extraction_batch_size)
        for i in range(0, patches.size(0), chunk):
            vectors.append(self._fre_vector(patches[i:i + chunk]))
        return torch.cat(vectors, dim=0)

    def _forward_fixed(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x01 = self._resize_to_patch(x01)
        features = self._fre_vector(x01)
        return self.cls_head(features), features

    def _forward_arbitrary(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        patches = self._patchify_batch(x01)
        b, l, c, h, w = patches.shape
        flat = patches.view(b * l, c, h, w)
        patch_features = self._vectors_in_chunks(flat).view(b, l, -1)
        features = self.patch_norm(self.patch_attention(patch_features))
        return self.cls_head(features), features

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x01 = self._to_01(x)
        if self.resolution_mode == "fixed":
            logits, features = self._forward_fixed(x01)
        else:
            logits, features = self._forward_arbitrary(x01)
        return {"logits": logits, "spai_features": features}


@MODELS.register_module(name="SPAIMFM")
class SPAIMFM(SPAIBase):
    """Masked frequency modeling pretraining on real images."""

    def __init__(
        self,
        backbone_name: str = "vit_base_patch16_224",
        pretrained: bool = True,
        feature_layers: Optional[Sequence[int]] = None,
        img_size: int = 224,
        masking_radius: int = 16,
        loss_gamma: float = 1.0,
        matrix_gamma: float = 1.0,
        patch_factor: int = 1,
        ave_spectrum: bool = False,
        with_matrix: bool = False,
        log_matrix: bool = False,
        batch_matrix: bool = False,
        input_is_normalized: bool = False,
        input_mean: Sequence[float] = IMAGENET_MEAN,
        input_std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__(input_is_normalized=input_is_normalized, input_mean=input_mean, input_std=input_std)
        self.img_size = int(img_size)
        self.masking_radius = int(masking_radius)
        self.backbone = TimmIntermediateViT(backbone_name, pretrained=pretrained, feature_layers=feature_layers)
        patch_size = self.backbone.patch_size
        self.patch_size = int(patch_size)
        self.decoder = nn.Linear(self.backbone.embed_dim, 3 * self.patch_size * self.patch_size)
        self.criterion = FrequencyLoss(
            loss_gamma=loss_gamma,
            matrix_gamma=matrix_gamma,
            patch_factor=patch_factor,
            ave_spectrum=ave_spectrum,
            with_matrix=with_matrix,
            log_matrix=log_matrix,
            batch_matrix=batch_matrix,
        )
        self.decoder.apply(_init_weights)

    def _resize_input(self, x01: torch.Tensor) -> torch.Tensor:
        if x01.shape[-2:] == (self.img_size, self.img_size):
            return x01
        return F.interpolate(x01, size=(self.img_size, self.img_size), mode="bicubic", align_corners=False)

    def _corrupt(self, x01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, h, w = x01.shape
        mask = generate_circular_mask(h, w, self.masking_radius, device=x01.device, dtype=torch.float32)
        low_freq, _ = filter_image_frequencies(x01, mask)
        return low_freq.clamp(0.0, 1.0).to(x01.dtype), mask

    def _decode(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, l, _ = tokens.shape
        patch = self.patch_size
        gh = height // patch
        gw = width // patch
        if l != gh * gw:
            raise RuntimeError(f"Token grid mismatch: got {l} tokens, expected {gh * gw}.")
        patches = self.decoder(tokens)
        patches = patches.view(b, gh, gw, 3, patch, patch)
        return patches.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, 3, gh * patch, gw * patch)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x01 = self._resize_input(self._to_01(x))
        corrupted, mask = self._corrupt(x01)
        features = self.backbone(self._normalize(corrupted))
        tokens = features[:, -1]
        rec = self._decode(tokens, x01.shape[-2], x01.shape[-1])
        loss_map = self.criterion(rec, x01)
        weight = (1.0 - mask).unsqueeze(1)
        loss = (loss_map * weight).sum() / (weight.sum().clamp_min(1e-6) * x01.size(1) * loss_map.size(1))
        return {"loss": loss, "reconstruction": rec, "corrupted": corrupted}
