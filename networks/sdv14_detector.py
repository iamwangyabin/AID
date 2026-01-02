
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import MODELS

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    np = None  # type: ignore

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None  # type: ignore

try:
    from diffusers import DDPMScheduler, StableDiffusionPipeline
except Exception as e:  # pragma: no cover
    DDPMScheduler = None  # type: ignore
    StableDiffusionPipeline = None  # type: ignore

try:
    from transformers import AutoImageProcessor, CLIPModel
    from transformers import pipeline as hf_pipeline
except Exception as e:  # pragma: no cover
    AutoImageProcessor = None  # type: ignore
    CLIPModel = None  # type: ignore
    hf_pipeline = None  # type: ignore


def _normalize_batch(batch: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """L2-normalize each element in a batch, regardless of element dimensionality."""
    dims = tuple(range(1, batch.dim()))
    norms = torch.norm(batch, p=2, dim=dims, keepdim=True)
    return batch / (norms + float(epsilon))


def _resize_then_center_crop(x: torch.Tensor, size: int, interpolation: str = "bicubic") -> torch.Tensor:
    """
    Resize to (size+3, size+3) then center-crop to (size, size), matching the provided reference.
    x: [B,3,H,W] or [3,H,W]
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    target = int(size)
    resize_to = target + 3

    x = F.interpolate(
        x,
        size=(resize_to, resize_to),
        mode=interpolation,
        align_corners=False if interpolation in {"bilinear", "bicubic"} else None,
    )

    top = (x.size(-2) - target) // 2
    left = (x.size(-1) - target) // 2
    x = x[:, :, top : top + target, left : left + target]

    if squeeze:
        x = x.squeeze(0)
    return x


def _postprocess_to_uint8_hwc(x: torch.Tensor, size: int, interpolation: str = "bicubic", do_resize: bool = True) -> "np.ndarray":
    """
    x: [B,3,H,W] in (approximately) [-1,1]
    returns: uint8 numpy [B,size,size,3]
    """
    if np is None:
        raise ImportError("numpy is required for SDv14CriterionModel postprocessing.")

    if do_resize:
        x = _resize_then_center_crop(x, size=size, interpolation=interpolation)

    x = (x / 2.0 + 0.5).clamp(0, 1) * 255.0
    x = x.detach().cpu()
    x = x.permute(0, 2, 3, 1).float().numpy()
    x = np.round(x).astype("uint8")
    return x


def _numpy_chunk_first_dim(arr: "np.ndarray", num_chunks: int) -> List["np.ndarray"]:
    """Split a numpy array into `num_chunks` chunks along axis=0, keeping approximate equality."""
    if np is None:
        raise ImportError("numpy is required for SDv14CriterionModel chunking.")

    chunk_size = arr.shape[0] // num_chunks
    remainder = arr.shape[0] % num_chunks
    indices = np.cumsum([0] + [chunk_size + 1 if i < remainder else chunk_size for i in range(num_chunks)])
    return list(np.split(arr, indices[1:-1], axis=0))


def _to_pil(img_uint8_hwc: torch.Tensor) -> "Image.Image":
    if Image is None:
        raise ImportError("PIL is required for SDv14CriterionModel.")
    if img_uint8_hwc.dtype != torch.uint8:
        raise ValueError("Expected uint8 HWC tensor for PIL conversion.")
    arr = img_uint8_hwc.detach().cpu().numpy()
    return Image.fromarray(arr, mode="RGB")


@MODELS.register_module()
class SDv14CriterionModel(nn.Module):
    """
    Training-free detector based on Stable Diffusion v1.4 denoiser behavior + CLIP geometry.

    Interface:
      - forward(x) returns dict with key 'logits' so it works with [`utils.validate_plain()`](utils/validate.py:61).

    Notes:
      - This model is heavy: StableDiffusion v1.4 + CLIP ViT-L/14 (+ optional captioning model).
      - By default it lazy-loads models on the first forward pass (so import/config parsing is fast).
      - Recommended dataloader transform: Resize(siz) + CenterCrop(siz) + ToTensor() (NO Normalize).
    """

    def __init__(
        self,
        siz: int = 512,
        num_noise: int = 8,
        time_frac: float = 0.01,
        epsilon_reg: float = 1e-8,
        threshold: float = 1.0,
        temperature: float = 0.1,
        interpolation: str = "bicubic",
        sd_model_id: str = "CompVis/stable-diffusion-v1-4",
        clip_model_id: str = "openai/clip-vit-large-patch14",
        # If provided, used for auto-captioning; otherwise you should provide prompts.
        caption_model_id: Optional[str] = "llava-hf/llava-1.5-7b-hf",
        prompts: Optional[Sequence[str]] = None,
        return_terms: bool = False,
        verbose: bool = False,
        lazy_init: bool = True,
        max_vae_decode_batch: int = 16,
        caption_max_new_tokens: int = 76,
    ) -> None:
        super().__init__()

        self.siz = int(siz)
        self.num_noise = int(num_noise)
        self.time_frac = float(time_frac)
        self.epsilon_reg = float(epsilon_reg)
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.interpolation = str(interpolation)

        self.sd_model_id = str(sd_model_id)
        self.clip_model_id = str(clip_model_id)
        self.caption_model_id = caption_model_id if caption_model_id is None else str(caption_model_id)

        self.prompts = list(prompts) if prompts is not None else None
        self.return_terms = bool(return_terms)
        self.verbose = bool(verbose)
        self.lazy_init = bool(lazy_init)
        self.max_vae_decode_batch = int(max_vae_decode_batch)
        self.caption_max_new_tokens = int(caption_max_new_tokens)

        # Lazy-loaded components
        self._initialized = False

        # These will be set in _lazy_init()
        self.unet = None
        self.vae = None
        self.text_encoder = None
        self.tokenizer = None
        self.scheduler = None

        self.clip = None
        self.processor = None

        self._caption_pipe = None
        self._cos = None

        if not self.lazy_init:
            # Initialize on CPU; will move when `.cuda()` called and/or in forward().
            self._lazy_init(device=torch.device("cpu"))

    def _require_deps(self) -> None:
        if StableDiffusionPipeline is None or DDPMScheduler is None:
            raise ImportError(
                "Missing dependency `diffusers`. Please install diffusers, accelerate, safetensors.\n"
                "Example: pip install diffusers accelerate safetensors"
            )
        if AutoImageProcessor is None or CLIPModel is None:
            raise ImportError(
                "Missing dependency `transformers`. Please install transformers.\n"
                "Example: pip install transformers"
            )
        if np is None:
            raise ImportError("Missing dependency `numpy`.")
        if Image is None:
            raise ImportError("Missing dependency `Pillow`.")

    def _lazy_init(self, device: torch.device) -> None:
        if self._initialized:
            return

        self._require_deps()

        torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        # Stable Diffusion v1.4 components
        pipe = StableDiffusionPipeline.from_pretrained(self.sd_model_id, torch_dtype=torch_dtype)
        self.unet = pipe.unet.eval()
        self.vae = pipe.vae.eval()
        self.text_encoder = pipe.text_encoder.eval()
        self.tokenizer = pipe.tokenizer
        self.scheduler = DDPMScheduler.from_pretrained(self.sd_model_id, subfolder="scheduler")

        for m in [self.unet, self.vae, self.text_encoder]:
            for p in m.parameters():
                p.requires_grad = False

        # CLIP
        self.processor = AutoImageProcessor.from_pretrained(self.clip_model_id)
        self.clip = CLIPModel.from_pretrained(self.clip_model_id).eval()
        for p in self.clip.parameters():
            p.requires_grad = False

        self._cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        # Move modules to device
        self.unet.to(device)
        self.vae.to(device)
        self.text_encoder.to(device)
        self.clip.to(device)
        self._cos.to(device)

        self._initialized = True

    def _get_caption_pipe(self, device: torch.device):
        if self.caption_model_id is None:
            return None
        if self._caption_pipe is not None:
            return self._caption_pipe
        if hf_pipeline is None:
            raise ImportError("transformers.pipeline is required for captioning (transformers not installed).")

        # transformers pipeline expects int device: -1 for cpu, 0.. for cuda
        if device.type == "cuda":
            dev_index = device.index if device.index is not None else 0
        else:
            dev_index = -1

        self._caption_pipe = hf_pipeline("image-to-text", model=self.caption_model_id, device=dev_index)
        return self._caption_pipe

    def _prep_input_uint8_hwc_list(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x: [B,3,H,W] float/uint8 in [0,1] or [0,255]
        returns list of uint8 [H,W,3] on CPU
        """
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"Expected input [B,3,H,W], got {tuple(x.shape)}")

        x_f = x.float()
        if x_f.max() > 1.5:
            x01 = (x_f / 255.0).clamp(0.0, 1.0)
        else:
            x01 = x_f.clamp(0.0, 1.0)

        x_u8 = torch.round(x01 * 255.0).clamp(0, 255).to(torch.uint8)
        # Convert to list of HWC tensors on CPU
        return [img.permute(1, 2, 0).contiguous().cpu() for img in x_u8]

    def _preprocess_sd_images(self, images_u8_hwc: List[torch.Tensor], device: torch.device) -> torch.Tensor:
        """
        Convert list of uint8 HWC into SD input tensor:
          [B,3,siz,siz] float16/float32 in [-1,1]
        """
        imgs = [img.permute(2, 0, 1).to(device) for img in images_u8_hwc]  # CHW uint8
        imgs_f = torch.stack([im.float() for im in imgs], dim=0)

        # resize+crop in float (still 0..255)
        imgs_f = _resize_then_center_crop(imgs_f, size=self.siz, interpolation=self.interpolation)

        # scale to [-1,1]
        imgs_f = 2.0 * (imgs_f / 255.0) - 1.0

        # use fp16 on cuda, fp32 on cpu
        if device.type == "cuda":
            imgs_f = imgs_f.half()
        return imgs_f

    def _get_prompts(self, images_u8_hwc: List[torch.Tensor], device: torch.device) -> List[str]:
        n = len(images_u8_hwc)
        if self.prompts is not None:
            if len(self.prompts) == 1:
                return list(self.prompts) * n
            if len(self.prompts) != n:
                raise ValueError(f"prompts length must be 1 or batch_size; got {len(self.prompts)} vs {n}.")
            return list(self.prompts)

        # Auto-caption
        cap = self._get_caption_pipe(device)
        if cap is None:
            raise ValueError("No prompts provided and caption_model_id is None; cannot auto-caption.")

        prompts: List[str] = []
        for img in images_u8_hwc:
            pil = _to_pil(img)
            out = cap(
                pil,
                prompt="<image>\nUSER: Generate a caption for the image that contains only facts and detailed\nASSISTANT:",
                generate_kwargs={"max_new_tokens": self.caption_max_new_tokens},
            )[0]
            text = out.get("generated_text", "")
            if "ASSISTANT:" in text:
                text = text.split("ASSISTANT:", 1)[1]
            prompts.append(text.strip())
        return prompts

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
          x: [B,3,H,W] tensor, preferably after ToTensor() only (no Normalize).

        Returns:
          dict with:
            - logits: [B,1] (higher => more likely fake)
            - criterion: [B,1] (raw score from the method)
            - (optional) bias/kappa/D: [B,1] if return_terms=True
        """
        device = x.device
        self._lazy_init(device=device)

        assert self.unet is not None
        assert self.vae is not None
        assert self.text_encoder is not None
        assert self.tokenizer is not None
        assert self.scheduler is not None
        assert self.clip is not None
        assert self.processor is not None
        assert self._cos is not None

        if self.num_noise <= 0:
            raise ValueError("num_noise must be positive.")

        # 1) Prepare images/prompts
        images_u8_hwc = self._prep_input_uint8_hwc_list(x)  # list of uint8 HWC (CPU)
        num_images = len(images_u8_hwc)
        prompts = self._get_prompts(images_u8_hwc, device=device)

        images_sd = self._preprocess_sd_images(images_u8_hwc, device=device)  # [B,3,siz,siz] in [-1,1]

        # 2) Text embeddings (repeat each prompt K times)
        expanded_prompts: List[str] = []
        for p in prompts:
            expanded_prompts.extend([p] * self.num_noise)

        text_tokens = self.tokenizer(
            expanded_prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = text_tokens.input_ids.to(device)
        text_emb = self.text_encoder(input_ids).last_hidden_state  # [B*K, 77, D]

        # 3) Encode images to latent space
        latents = self.vae.encode(images_sd).latent_dist.sample()
        latents = latents * float(self.vae.config.scaling_factor)  # [B,4,64,64] (for siz=512)

        latents = latents.repeat_interleave(self.num_noise, dim=0)  # [B*K,4,64,64]
        if device.type == "cuda":
            latents = latents.half()

        # 4) Spherical noise in latent space (Gaussian -> normalize -> scale by sqrt(d))
        gauss_noise = torch.randn_like(latents, device=device)
        if device.type == "cuda":
            gauss_noise = gauss_noise.half()

        spherical_noise = _normalize_batch(gauss_noise, epsilon=self.epsilon_reg)
        d_lat = int(torch.tensor(latents.shape[1:]).prod().item())
        spherical_noise = spherical_noise * math.sqrt(float(d_lat))

        # 5) Choose timestep and add noise
        t_float = self.time_frac * float(self.scheduler.config.num_train_timesteps)
        t_int = int(round(t_float))
        t_int = max(0, min(int(self.scheduler.config.num_train_timesteps) - 1, t_int))
        timestep = torch.full((latents.shape[0],), t_int, device=device, dtype=torch.long)

        noisy_latents = self.scheduler.add_noise(
            original_samples=latents,
            noise=spherical_noise,
            timesteps=timestep,
        )
        if device.type == "cuda":
            noisy_latents = noisy_latents.half()

        if self.verbose:
            alpha_t = self.scheduler.alphas_cumprod[int(timestep[0].item())]
            print(f"\ntimestep: {int(timestep[0].item())}, alpha_t: {float(alpha_t)}")
            print(f"dimension of latent space: {d_lat}")

        # 6) Predict noise with UNet
        noise_pred = self.unet(noisy_latents, timestep, encoder_hidden_states=text_emb)[0]
        noise_pred = noise_pred / float(self.vae.config.scaling_factor)

        # Free some memory before decoding
        del noisy_latents, gauss_noise, latents, images_sd

        # 7) Decode predicted noise & spherical noise to image space (with batching)
        sub_bs = max(1, int(self.max_vae_decode_batch))
        decoded_noise_parts: List[torch.Tensor] = []
        decoded_sph_parts: List[torch.Tensor] = []

        total = noise_pred.size(0)
        for start in range(0, total, sub_bs):
            end = min(start + sub_bs, total)
            decoded_noise_parts.append(self.vae.decode(noise_pred[start:end], return_dict=False)[0])
            decoded_sph_parts.append(self.vae.decode(spherical_noise[start:end], return_dict=False)[0])

        decoded_noise = torch.cat(decoded_noise_parts, dim=0)  # [B*K,3,H,W] in [-1,1]
        decoded_spherical = torch.cat(decoded_sph_parts, dim=0)

        # 8) Convert decoded tensors to uint8 HWC numpy for CLIP processor
        decoded_noise_u8 = _postprocess_to_uint8_hwc(decoded_noise, size=self.siz, interpolation=self.interpolation, do_resize=True)
        decoded_spherical_u8 = _postprocess_to_uint8_hwc(decoded_spherical, size=self.siz, interpolation=self.interpolation, do_resize=True)

        decoded_noise_chunks = _numpy_chunk_first_dim(decoded_noise_u8, num_chunks=num_images)
        decoded_spherical_chunks = _numpy_chunk_first_dim(decoded_spherical_u8, num_chunks=num_images)

        # 9) CLIP scoring per image
        criterion_list: List[float] = []
        bias_list: List[float] = []
        kappa_list: List[float] = []
        d_list: List[float] = []

        for img_u8_hwc, dec_noise_np, dec_sph_np in zip(images_u8_hwc, decoded_noise_chunks, decoded_spherical_chunks):
            # Original image CLIP feature
            img_in = self.processor(images=img_u8_hwc.numpy(), return_tensors="pt").to(device)
            img_feat = self.clip.get_image_features(**img_in)

            # Decoded UNet prediction CLIP features (K images)
            img_d_in = self.processor(images=dec_noise_np, return_tensors="pt").to(device)
            img_d_feat = self.clip.get_image_features(**img_d_in)

            # Decoded spherical noise CLIP features (K images)
            img_s_in = self.processor(images=dec_sph_np, return_tensors="pt").to(device)
            img_s_feat = self.clip.get_image_features(**img_s_in)

            # bias: cos(img, decoded_noise_k)
            bias_vec = self._cos(img_feat.expand_as(img_d_feat), img_d_feat)  # [K]
            # kappa: cos(decoded_noise_k, decoded_spherical_k)
            kappa_vec = self._cos(img_d_feat, img_s_feat)  # [K]
            # D: ||decoded_noise_k||_2
            D_vec = torch.norm(img_d_feat, p=2, dim=1)  # [K]

            bias_mean = float(bias_vec.mean().detach().cpu())
            kappa_mean = float(kappa_vec.mean().detach().cpu())
            D_mean = float(D_vec.mean().detach().cpu())

            d_clip = int(img_feat.shape[1])
            sqrt_d_clip = math.sqrt(float(d_clip))

            # As in your reference: sqrt(d), 1, -1 coefficients
            criterion = 1.0 + (sqrt_d_clip * bias_mean - D_mean + kappa_mean) / (sqrt_d_clip + 2.0)

            criterion_list.append(float(criterion))
            bias_list.append(bias_mean)
            kappa_list.append(kappa_mean)
            d_list.append(D_mean)

        criterion_t = torch.tensor(criterion_list, device=device, dtype=torch.float32).view(-1, 1)

        # 10) Calibrate to logits (same convention as other detectors in this repo)
        temp = max(self.temperature, 1e-6)
        logits = (criterion_t - float(self.threshold)) / float(temp)

        out: Dict[str, torch.Tensor] = {
            "logits": logits,
            "criterion": criterion_t,
        }

        if self.return_terms:
            out["bias"] = torch.tensor(bias_list, device=device, dtype=torch.float32).view(-1, 1)
            out["kappa"] = torch.tensor(kappa_list, device=device, dtype=torch.float32).view(-1, 1)
            out["D"] = torch.tensor(d_list, device=device, dtype=torch.float32).view(-1, 1)

        return out