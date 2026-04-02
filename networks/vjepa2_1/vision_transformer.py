import math
from functools import partial

import torch
import torch.nn as nn

from networks.vjepa2_1.modules import Block
from networks.vjepa2_1.patch_embed import PatchEmbed, PatchEmbed3D


def apply_masks(x, masks):
    all_x = []
    for mask in masks:
        mask_keep = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x.append(torch.gather(x, dim=1, index=mask_keep))
    return torch.cat(all_x, dim=0)


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        lower = norm_cdf((a - mean) / std)
        upper = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        use_silu=False,
        wide_silu=True,
        use_sdpa=True,
        use_activation_checkpointing=False,
        is_causal=False,
        use_rope=False,
        init_type="default",
        handle_nonsquare_inputs=True,
        img_temporal_dim_size=None,
        n_registers=0,
        has_cls_first=False,
        interpolate_rope=False,
        modality_embedding=True,
        n_output_distillation=4,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.out_layers = out_layers
        self.init_type = init_type
        self.handle_nonsquare_inputs = handle_nonsquare_inputs
        self.img_temporal_dim_size = img_temporal_dim_size

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.is_video = num_frames > 1
        self.use_activation_checkpointing = use_activation_checkpointing
        self.uniform_power = uniform_power
        self.use_rope = use_rope
        self.init_std = init_std
        self.attn_out = False
        self.cls_token = None
        self.return_hierarchical = False

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        if self.is_video:
            self.patch_embed = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (
                (num_frames // tubelet_size)
                * (img_size[0] // patch_size)
                * (img_size[1] // patch_size)
            )
        else:
            self.patch_embed = PatchEmbed(
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        if self.img_temporal_dim_size is not None:
            self.patch_embed_img = PatchEmbed3D(
                patch_size=patch_size,
                tubelet_size=1,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed_img = None

        self.blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=img_size[0] // patch_size,
                    grid_depth=num_frames // tubelet_size,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    use_sdpa=use_sdpa,
                    is_causal=is_causal,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    n_registers=n_registers,
                    has_cls_first=has_cls_first,
                    interpolate_rope=interpolate_rope,
                    patch_size=patch_size,
                )
                for i in range(depth)
            ]
        )

        self.apply(self._init_weights)
        self._rescale_blocks()

        if depth == 12:
            self.hierarchical_layers = [2, 5, 8, 11]
            self.out_layers_distillation = [11] if n_output_distillation == 1 else [2, 5, 8, 11]
        elif depth == 24:
            self.hierarchical_layers = [5, 11, 17, 23]
            self.out_layers_distillation = [23] if n_output_distillation == 1 else [5, 11, 17, 23]
        elif depth == 40:
            self.hierarchical_layers = [9, 19, 29, 39]
            self.out_layers_distillation = [39] if n_output_distillation == 1 else [9, 19, 29, 39]
        elif depth == 48:
            self.hierarchical_layers = [11, 23, 37, 47]
            self.out_layers_distillation = [47] if n_output_distillation == 1 else [11, 23, 37, 47]
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        self.norms_block = nn.ModuleList(
            [norm_layer(embed_dim) for _ in range(len(self.hierarchical_layers))]
        )

        self.modality_embedding = False
        if modality_embedding:
            self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.video_mod_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.normal_(self.img_mod_embed, std=1e-6)
            nn.init.normal_(self.video_mod_embed, std=1e-6)
            self.modality_embedding = True

    def _init_weights(self, module):
        if isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
            return

        if self.init_type == "default":
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                trunc_normal_(module.weight, std=self.init_std)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        elif self.init_type == "xavier_uniform":
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        elif self.init_type == "xavier_normal":
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        else:
            raise ValueError(f"Unknown init type {self.init_type}")

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data if hasattr(layer.mlp, "fc2") else layer.mlp.fc3.weight.data, layer_id + 1)

    def check_temporal_dim(self, shape):
        return self.img_temporal_dim_size is not None and shape[2] == self.img_temporal_dim_size

    def forward(self, x, masks=None, training=False):
        if masks is not None and not isinstance(masks, list):
            masks = [masks]

        if x.ndim == 4:
            _, _, H, W = x.shape
            T = 1
        elif x.ndim == 5:
            _, _, T, H, W = x.shape
            T = T // 1 if self.check_temporal_dim(x.shape) else T // self.tubelet_size
        else:
            raise ValueError(f"Unsupported input shape: {tuple(x.shape)}")

        H_patches = H // self.patch_size
        W_patches = W // self.patch_size
        if not self.handle_nonsquare_inputs:
            T = H_patches = W_patches = None

        if self.check_temporal_dim(x.shape):
            x = self.patch_embed_img(x)
            mode = "img"
            if self.modality_embedding:
                x = x + self.img_mod_embed.repeat(x.shape[0], 1, 1)
        else:
            x = self.patch_embed(x)
            mode = "video"
            if self.modality_embedding:
                x = x + self.video_mod_embed.repeat(x.shape[0], 1, 1)

        if masks is not None:
            x = apply_masks(x, masks)
            masks = torch.cat(masks, dim=0)

        outs = []
        hier = []
        for i, block in enumerate(self.blocks):
            if self.use_activation_checkpointing:
                x, attn = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    use_reentrant=False,
                    return_attn=self.attn_out,
                    mode=mode,
                )
            else:
                x, attn = block(
                    x,
                    mask=masks,
                    T=T,
                    H_patches=H_patches,
                    W_patches=W_patches,
                    return_attn=self.attn_out,
                    mode=mode,
                )

            if self.out_layers is not None and i in self.out_layers:
                out_idx = self.hierarchical_layers.index(i)
                outs.append(self.norms_block[out_idx](x))

            if i in self.out_layers_distillation:
                out_idx = self.hierarchical_layers.index(i)
                hier.append(self.norms_block[out_idx](x))

        if self.out_layers is not None:
            return outs

        if training or self.return_hierarchical:
            return torch.cat(hier, dim=2)
        return self.norms_block[-1](x)


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_giant_xformers(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_gigantic_xformers(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=1664,
        depth=48,
        num_heads=26,
        mlp_ratio=64 / 13,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
