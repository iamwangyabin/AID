import math
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.registry import MODELS


def _resolve_norm_layer(norm_layer: Literal["batch", "instance"]):
    if norm_layer == "batch":
        return nn.BatchNorm2d
    if norm_layer == "instance":
        return nn.InstanceNorm2d
    raise ValueError(f"Unknown norm layer: {norm_layer}")


def _resnet50_weights(pretrained: bool):
    if not pretrained:
        return None
    return torchvision.models.ResNet50_Weights.IMAGENET1K_V2


def _make_fire_resnet(
    mode: Literal["frq"] = "frq",
    norm_layer: Literal["batch", "instance"] = "instance",
    pretrained: bool = True,
) -> nn.Module:
    if mode != "frq":
        raise ValueError("FIRE uses the frequency-guided 6-channel backend; set mode='frq'.")

    norm = _resolve_norm_layer(norm_layer)
    weights = _resnet50_weights(pretrained)

    if norm is nn.InstanceNorm2d and pretrained:
        model = torchvision.models.resnet50(num_classes=1000, weights=None, norm_layer=norm)
        model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True), strict=False)
    else:
        model = torchvision.models.resnet50(num_classes=1000, weights=weights, norm_layer=norm)

    model.conv1.weight = nn.Parameter(torch.cat([model.conv1.weight * 0.25] * 2, dim=1))
    model.conv1.in_channels = 6
    model.fc = nn.Linear(2048, 1)
    nn.init.normal_(model.fc.weight.data, 0.0, 0.02)
    if model.fc.bias is not None:
        nn.init.zeros_(model.fc.bias.data)
    return model


def _resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported VAE dtype: {dtype}")


def _retrieve_latents(encoder_output, latent_mode: str) -> torch.Tensor:
    latent_dist = getattr(encoder_output, "latent_dist", None)
    if latent_dist is None:
        if hasattr(encoder_output, "latents"):
            return encoder_output.latents
        if isinstance(encoder_output, (tuple, list)):
            return encoder_output[0]
        raise RuntimeError("Unexpected VAE encoder output; cannot retrieve latents.")

    latent_mode = str(latent_mode).lower()
    if latent_mode == "sample":
        return latent_dist.sample()
    if latent_mode in {"mode", "mean"}:
        return latent_dist.mode()
    raise ValueError(f"Unsupported latent_mode: {latent_mode}")


class ESPCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int) -> None:
        super().__init__()
        hidden_channels = channels // 2
        shuffled_channels = int(out_channels * (upscale_factor**2))
        self.bn = nn.BatchNorm2d(in_channels)
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
        self.sub_pixel_0 = nn.Sequential(
            nn.Conv2d(hidden_channels, shuffled_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )
        self.sub_pixel_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, shuffled_channels, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == hidden_channels:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                else:
                    fan = module.out_channels * module.weight.data[0][0].numel()
                    nn.init.normal_(module.weight.data, 0.0, math.sqrt(2 / fan))
                nn.init.zeros_(module.bias.data)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bn(x)
        x = self.feature_maps(x)
        return self.sub_pixel_0(x), self.sub_pixel_1(x)


class FFTFilter(nn.Module):
    def __init__(self, radiuslow: int = 40, radiushigh: int = 120, rows: int = 256, cols: int = 256):
        super().__init__()
        self.radiuslow = int(radiuslow)
        self.radiushigh = int(radiushigh)
        self.rows = int(rows)
        self.cols = int(cols)

        i_mask, r_i_mask = self._init_mask()
        self.register_buffer("i_mask", i_mask, persistent=True)
        self.register_buffer("r_i_mask", r_i_mask, persistent=True)
        self.mask_autoencoder = ESPCN(in_channels=3, out_channels=1, channels=64, upscale_factor=1)

    def _init_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones((1, self.rows, self.cols), dtype=torch.float32)
        crow, ccol = self.rows // 2, self.cols // 2
        y, x = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing="ij")
        radius_sq = (y - crow) ** 2 + (x - ccol) ** 2
        mask[:, radius_sq < self.radiuslow * self.radiuslow] = 0
        mask[:, radius_sq >= self.radiushigh * self.radiushigh] = 0
        return mask, 1.0 - mask

    @staticmethod
    def _normalize_like_official(x: torch.Tensor) -> torch.Tensor:
        denom = (x.amax() - x.amin()).clamp_min(1e-6)
        return x / denom

    def middle_pass_filter(self, image: torch.Tensor):
        freq_image = torch.fft.fftn(image * 255.0, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))

        log_freq = (20.0 * torch.log(torch.abs(freq_image) + 1e-7)) / 255.0
        mask_mid_frq, mask_mid_filtered = self.mask_autoencoder(log_freq)

        middle_freq = freq_image * mask_mid_frq.to(freq_image.dtype)
        middle_freq = torch.fft.ifftshift(middle_freq, dim=(-2, -1))
        middle_freq_image = torch.abs(torch.fft.ifftn(middle_freq, dim=(-2, -1)))
        middle_freq_image = self._normalize_like_official(middle_freq_image)

        middle_filtered = freq_image * mask_mid_filtered.to(freq_image.dtype)
        middle_filtered = torch.fft.ifftshift(middle_filtered, dim=(-2, -1))
        middle_filtered_image = torch.abs(torch.fft.ifftn(middle_filtered, dim=(-2, -1)))
        middle_filtered_image = self._normalize_like_official(middle_filtered_image)

        return middle_freq_image, middle_filtered_image, mask_mid_frq.float(), mask_mid_filtered.float()

    def forward(self, image: torch.Tensor):
        return self.middle_pass_filter(image)


@MODELS.register_module(name=["FIREModel", "FIRE"])
class FIREModel(nn.Module):
    """
    FIRE: Robust Detection of Diffusion-Generated Images via Frequency-Guided
    Reconstruction Error (CVPR 2025), adapted to this framework.

    Expected input is RGB tensors in [0, 1] with no ImageNet/CLIP normalization.
    A [-1, 1] tensor is also accepted and converted back to [0, 1].
    """

    def __init__(
        self,
        mode: Literal["frq"] = "frq",
        norm_layer: Literal["batch", "instance"] = "instance",
        pretrained: bool = True,
        radiuslow: int = 40,
        radiushigh: int = 120,
        image_size: int = 256,
        resize_input: bool = True,
        vae_model_name: str = "runwayml/stable-diffusion-v1-5",
        vae_subfolder: Optional[str] = "vae",
        vae_dtype: str = "auto",
        latent_mode: Literal["sample", "mode", "mean"] = "sample",
        freeze_vae: bool = True,
        input_mean: Optional[Sequence[float]] = None,
        input_std: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.resize_input = bool(resize_input)
        self.latent_mode = str(latent_mode)
        self.freeze_vae = bool(freeze_vae)
        self.vae_dtype = _resolve_dtype(vae_dtype)

        if input_mean is not None and input_std is not None:
            mean = torch.tensor(input_mean, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(input_std, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("_input_mean", mean, persistent=False)
            self.register_buffer("_input_std", std, persistent=False)
        else:
            self._input_mean = None
            self._input_std = None

        try:
            from diffusers import AutoencoderKL
        except ImportError as exc:
            raise ImportError("FIREModel requires diffusers. Install it with `pip install diffusers`.") from exc

        vae_kwargs = {"torch_dtype": self.vae_dtype}
        if vae_subfolder:
            vae_kwargs["subfolder"] = vae_subfolder
        self.vae = AutoencoderKL.from_pretrained(vae_model_name, **vae_kwargs)
        self.decode_dtype = next(self.vae.parameters()).dtype
        if self.freeze_vae:
            for param in self.vae.parameters():
                param.requires_grad = False
            self.vae.eval()

        self.resnet = _make_fire_resnet(mode=mode, norm_layer=norm_layer, pretrained=pretrained)
        self.fft_filter_module = FFTFilter(
            radiuslow=radiuslow,
            radiushigh=radiushigh,
            rows=self.image_size,
            cols=self.image_size,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vae:
            self.vae.eval()
        return self

    def _prep_pixel_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype != torch.float32:
            x = x.float()

        if self._input_mean is not None and self._input_std is not None:
            x = x * self._input_std + self._input_mean

        xmin = float(x.detach().amin())
        xmax = float(x.detach().amax())
        if xmin >= 0.0 and xmax > 1.5:
            x = x / 255.0
        elif xmin < -0.1 and xmax <= 1.5:
            x = (x + 1.0) * 0.5
        elif xmin < -0.1 and xmax > 1.5:
            raise ValueError(
                "FIRE expects unnormalized [0,1] tensors or [-1,1] tensors. "
                "Remove Normalize from the FIRE config, or pass matching input_mean/input_std."
            )

        x = x.clamp(0.0, 1.0)
        if x.shape[1] != 3:
            raise ValueError(f"FIRE expects RGB input with 3 channels, got C={x.shape[1]}.")
        if self.resize_input and x.shape[-2:] != (self.image_size, self.image_size):
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return x

    def _vae_reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        x_vae = x.to(dtype=self.decode_dtype)
        latents = _retrieve_latents(self.vae.encode(x_vae), self.latent_mode)
        return self.vae.decode(latents.to(dtype=self.decode_dtype), return_dict=False)[0].to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._prep_pixel_tensor(x)
        middle_freq_image, middle_filtered_image, mask_mid_frq, mask_mid_filtered = self.fft_filter_module(x)

        reconstructions_x = self._vae_reconstruct(x)
        reconstructions_middle_filtered = self._vae_reconstruct(middle_filtered_image)

        raw_reconstruction_delta = torch.abs(reconstructions_x - x)
        filtered_reconstruction_delta = torch.abs(reconstructions_middle_filtered - x)
        logits = self.resnet(torch.cat([raw_reconstruction_delta, filtered_reconstruction_delta], dim=1))

        return {
            "logits": logits,
            "middle_freq_image": middle_freq_image,
            "middle_filtered_image": middle_filtered_image,
            "raw_reconstruction_delta": raw_reconstruction_delta,
            "filtered_reconstruction_delta": filtered_reconstruction_delta,
            "mask_mid_frq": mask_mid_frq,
            "mask_mid_filtered": mask_mid_filtered,
        }
