import dataclasses
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from . import filters
from . import utils
from .vision_transformer import build_vit


class PatchBasedMFViT(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        features_processor: "FrequencyRestorationEstimator",
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_patch_size: int,
        img_patch_stride: int,
        cls_vector_dim: int,
        num_heads: int,
        attn_embed_dim: int,
        dropout: float = 0.0,
        frozen_backbone: bool = True,
        minimum_patches: int = 0,
    ) -> None:
        super().__init__()

        self.mfvit = MFViT(
            vit,
            features_processor,
            None,
            masking_radius,
            img_patch_size,
            frozen_backbone=frozen_backbone,
        )

        self.img_patch_size = img_patch_size
        self.img_patch_stride = img_patch_stride
        self.minimum_patches = minimum_patches
        self.cls_vector_dim = cls_vector_dim

        dim_head = attn_embed_dim // num_heads
        self.heads = num_heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_kv = nn.Linear(cls_vector_dim, attn_embed_dim * 2, bias=False)
        self.patch_aggregator = nn.Parameter(
            torch.zeros((num_heads, 1, attn_embed_dim // num_heads))
        )
        nn.init.trunc_normal_(self.patch_aggregator, std=0.02)
        self.to_out = nn.Sequential(
            nn.Linear(attn_embed_dim, cls_vector_dim, bias=False),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(cls_vector_dim)
        self.cls_head = cls_head

        self.apply(_init_weights)

    def patches_attention(self, x: torch.Tensor) -> torch.Tensor:
        aggregator = self.patch_aggregator.unsqueeze(0).expand(x.size(0), -1, -1, -1)
        kv = self.to_kv(x).chunk(2, dim=-1)
        k, v = kv
        b, n, _ = k.shape
        k = k.view(b, n, self.heads, -1).transpose(1, 2)
        v = v.view(b, n, self.heads, -1).transpose(1, 2)
        dots = torch.matmul(aggregator, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(b, 1, -1)
        x = self.to_out(x)
        x = x.squeeze(dim=1)
        return x

    def forward(self, x: torch.Tensor, feature_extraction_batch_size: Optional[int] = None) -> torch.Tensor:
        x = self.forward_batch(x, feature_extraction_batch_size)
        return x

    def forward_batch(self, x: torch.Tensor, feature_extraction_batch_size: Optional[int]) -> torch.Tensor:
        x = utils.patchify_image(
            x,
            (self.img_patch_size, self.img_patch_size),
            (self.img_patch_stride, self.img_patch_stride),
        )

        patch_features = []
        for i in range(x.size(1)):
            patch_features.append(self.mfvit(x[:, i]))
        x = torch.stack(patch_features, dim=1)

        x = self.patches_attention(x)
        x = self.norm(x)
        if self.cls_head is not None:
            x = self.cls_head(x)
        return x


class MFViT(nn.Module):
    def __init__(
        self,
        vit: nn.Module,
        features_processor: "FrequencyRestorationEstimator",
        cls_head: Optional[nn.Module],
        masking_radius: int,
        img_size: int,
        frozen_backbone: bool = True,
    ):
        super().__init__()
        self.vit = vit
        self.features_processor = features_processor
        self.cls_head = cls_head
        self.frozen_backbone = frozen_backbone

        self.frequencies_mask = nn.Parameter(
            filters.generate_circular_mask(img_size, masking_radius),
            requires_grad=False,
        )

        self.backbone_norm = transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        low_freq, hi_freq = filters.filter_image_frequencies(x.float(), self.frequencies_mask)

        low_freq = torch.clamp(low_freq, min=0.0, max=1.0).to(x.dtype)
        hi_freq = torch.clamp(hi_freq, min=0.0, max=1.0).to(x.dtype)

        x = self.backbone_norm(x)
        low_freq = self.backbone_norm(low_freq)
        hi_freq = self.backbone_norm(hi_freq)

        if self.frozen_backbone:
            with torch.no_grad():
                x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)
        else:
            x, low_freq, hi_freq = self._extract_features(x, low_freq, hi_freq)

        x = self.features_processor(x, low_freq, hi_freq)
        if self.cls_head is not None:
            x = self.cls_head(x)
        return x

    def _extract_features(
        self, x: torch.Tensor, low_freq: torch.Tensor, hi_freq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.vit(x)
        low_freq = self.vit(low_freq)
        hi_freq = self.vit(hi_freq)
        return x, low_freq, hi_freq


class FrequencyRestorationEstimator(nn.Module):
    def __init__(
        self,
        features_num: int,
        input_dim: int,
        proj_dim: int,
        proj_layers: int,
        patch_projection: bool = False,
        patch_projection_per_feature: bool = False,
        proj_last_layer_activation_type: Optional[str] = "gelu",
        original_image_features_branch: bool = False,
        dropout: float = 0.5,
        disable_reconstruction_similarity: bool = False,
    ):
        super().__init__()

        if proj_last_layer_activation_type == "gelu":
            proj_last_layer_activation = nn.GELU
        elif proj_last_layer_activation_type is None:
            proj_last_layer_activation = nn.Identity
        else:
            raise RuntimeError(
                f"Unsupported activation type: {proj_last_layer_activation_type}"
            )

        if patch_projection and patch_projection_per_feature:
            self.patch_projector: nn.Module = FeatureSpecificProjector(
                features_num,
                proj_layers,
                input_dim,
                proj_dim,
                proj_last_layer_activation,
                dropout=dropout,
            )
        elif patch_projection:
            self.patch_projector = Projector(
                proj_layers,
                input_dim,
                proj_dim,
                proj_last_layer_activation,
                dropout=dropout,
            )
        else:
            self.patch_projector = nn.Identity()

        self.original_features_processor = None
        if original_image_features_branch:
            self.original_features_processor = FeatureImportanceProjector(
                features_num, proj_dim, proj_dim, proj_layers, dropout=dropout
            )

        self.disable_reconstruction_similarity = disable_reconstruction_similarity
        if self.disable_reconstruction_similarity:
            if self.original_features_processor is None:
                raise RuntimeError(
                    "Reconstruction similarity cannot be disabled without the original branch."
                )

    def forward(
        self, x: torch.Tensor, low_freq: torch.Tensor, hi_freq: torch.Tensor
    ) -> torch.Tensor:
        orig = self.patch_projector(x)
        low_freq = self.patch_projector(low_freq)
        hi_freq = self.patch_projector(hi_freq)

        if self.disable_reconstruction_similarity:
            x = self.original_features_processor(orig)
        else:
            sim_x_low_freq = F.cosine_similarity(orig, low_freq, dim=-1)
            sim_x_hi_freq = F.cosine_similarity(orig, hi_freq, dim=-1)
            sim_low_freq_hi_freq = F.cosine_similarity(low_freq, hi_freq, dim=-1)

            sim_x_low_freq_mean = sim_x_low_freq.mean(dim=-1)
            sim_x_low_freq_std = sim_x_low_freq.std(dim=-1)
            sim_x_hi_freq_mean = sim_x_hi_freq.mean(dim=-1)
            sim_x_hi_freq_std = sim_x_hi_freq.std(dim=-1)
            sim_low_freq_hi_freq_mean = sim_low_freq_hi_freq.mean(dim=-1)
            sim_low_freq_hi_freq_std = sim_low_freq_hi_freq.std(dim=-1)

            x = torch.cat(
                [
                    sim_x_low_freq_mean,
                    sim_x_low_freq_std,
                    sim_x_hi_freq_mean,
                    sim_x_hi_freq_std,
                    sim_low_freq_hi_freq_mean,
                    sim_low_freq_hi_freq_std,
                ],
                dim=1,
            )

            if self.original_features_processor is not None:
                orig = self.original_features_processor(orig)
                x = torch.cat([x, orig], dim=1)

        return x


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
                Projector(
                    proj_layers,
                    input_dim,
                    proj_dim,
                    last_layer_activation,
                    dropout=dropout,
                )
                for _ in range(intermediate_features_num)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = []
        for i, projector in enumerate(self.projectors):
            projected.append(projector(x[:, i, :, :]))
        x = torch.stack(projected, dim=1)
        return x


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
        patch_proj_layers = [nn.Dropout(dropout)]
        for i in range(proj_layers):
            patch_proj_layers.extend(
                [
                    nn.Linear(input_dim if i == 0 else proj_dim, proj_dim),
                    nn.GELU() if i < proj_layers - 1 else last_layer_activation(),
                    nn.Dropout(dropout),
                ]
            )
        self.projector = nn.Sequential(*patch_proj_layers)
        self.norm2 = nn.LayerNorm(proj_dim) if output_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x)
        x = self.projector(x)
        x = self.norm2(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(
        self, input_dim: int, num_classes: int, mlp_ratio: int = 1, dropout: float = 0.5
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, input_dim * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * mlp_ratio, input_dim * mlp_ratio),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * mlp_ratio, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


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
        self.proj1 = Projector(proj_layers, 2 * proj_dim, proj_dim, input_norm=False, dropout=dropout)
        self.proj2 = Projector(proj_layers, proj_dim, proj_dim, input_norm=False, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = x.mean(dim=2)
        x_std = x.std(dim=2)
        x = torch.cat([x_mean, x_std], dim=-1)

        x = self.proj1(x)
        x = torch.softmax(self.alpha, dim=1) * x
        x = torch.sum(x, dim=1)
        x = self.proj2(x)

        return x


@dataclasses.dataclass
class _ConfigWrapper:
    DATA: object
    MODEL: object


def _init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def build_mf_vit(config):
    vit = build_vit(config)

    fre = FrequencyRestorationEstimator(
        features_num=len(config.MODEL.VIT.INTERMEDIATE_LAYERS),
        input_dim=config.MODEL.VIT.EMBED_DIM,
        proj_dim=config.MODEL.VIT.PROJECTION_DIM,
        proj_layers=config.MODEL.VIT.PROJECTION_LAYERS,
        patch_projection=config.MODEL.VIT.PATCH_PROJECTION,
        patch_projection_per_feature=config.MODEL.VIT.PATCH_PROJECTION_PER_FEATURE,
        proj_last_layer_activation_type=config.MODEL.FRE.PROJECTOR_LAST_LAYER_ACTIVATION_TYPE,
        original_image_features_branch=config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH,
        dropout=config.MODEL.SID_DROPOUT,
        disable_reconstruction_similarity=config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY,
    )

    cls_vector_dim = 6 * len(config.MODEL.VIT.INTERMEDIATE_LAYERS)
    if config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH and config.MODEL.FRE.DISABLE_RECONSTRUCTION_SIMILARITY:
        cls_vector_dim = config.MODEL.VIT.PROJECTION_DIM
    elif config.MODEL.FRE.ORIGINAL_IMAGE_FEATURES_BRANCH:
        cls_vector_dim += config.MODEL.VIT.PROJECTION_DIM

    cls_head = ClassificationHead(
        input_dim=cls_vector_dim,
        num_classes=config.MODEL.NUM_CLASSES if config.MODEL.NUM_CLASSES > 2 else 1,
        mlp_ratio=config.MODEL.CLS_HEAD.MLP_RATIO,
        dropout=config.MODEL.SID_DROPOUT,
    )

    if config.MODEL.RESOLUTION_MODE == "fixed":
        model = MFViT(
            vit,
            fre,
            cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_size=config.DATA.IMG_SIZE,
        )
    elif config.MODEL.RESOLUTION_MODE == "arbitrary":
        model = PatchBasedMFViT(
            vit,
            fre,
            cls_head,
            masking_radius=config.MODEL.FRE.MASKING_RADIUS,
            img_patch_size=config.DATA.IMG_SIZE,
            img_patch_stride=config.MODEL.PATCH_VIT.PATCH_STRIDE,
            cls_vector_dim=cls_vector_dim,
            attn_embed_dim=config.MODEL.PATCH_VIT.ATTN_EMBED_DIM,
            num_heads=config.MODEL.PATCH_VIT.NUM_HEADS,
            dropout=config.MODEL.SID_DROPOUT,
            minimum_patches=config.MODEL.PATCH_VIT.MINIMUM_PATCHES,
        )
    else:
        raise RuntimeError(f"Unsupported resolution mode: {config.MODEL.RESOLUTION_MODE}")

    return model
