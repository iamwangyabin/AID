import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

from utils.registry import MODELS

try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:  # pragma: no cover - OmegaConf is already a project dependency.
    DictConfig = None
    OmegaConf = None


def _as_plain_dict(value):
    if DictConfig is not None and isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _load_clip_model(model_name_or_path, local_files_only=False, cache_dir=None):
    kwargs = {"local_files_only": local_files_only}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return CLIPModel.from_pretrained(model_name_or_path, **kwargs)


def _extract_foren_proj_weight(weight_or_checkpoint):
    if torch.is_tensor(weight_or_checkpoint):
        return weight_or_checkpoint

    checkpoint = weight_or_checkpoint
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported GAPL stage-1 weight type: {type(weight_or_checkpoint)}")

    candidate_keys = (
        "foren_proj.weight",
        "model.foren_proj.weight",
        "module.foren_proj.weight",
        "module.model.foren_proj.weight",
    )
    for key in candidate_keys:
        if key in checkpoint:
            return checkpoint[key]

    for key, value in checkpoint.items():
        if key.endswith("foren_proj.weight"):
            return value

    raise KeyError("Could not find foren_proj.weight in GAPL stage-1 checkpoint.")


class _GAPLBase(nn.Module):
    def __init__(
        self,
        clip_model_name_or_path="openai/clip-vit-large-patch14",
        local_files_only=False,
        cache_dir=None,
        feature_dim=1024,
        foren_dim=128,
        num_classes=1,
        freeze_backbone=True,
        **kwargs,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.foren_dim = int(foren_dim)

        clip_model = _load_clip_model(
            clip_model_name_or_path,
            local_files_only=bool(local_files_only),
            cache_dir=cache_dir,
        )

        if freeze_backbone:
            for param in clip_model.parameters():
                param.requires_grad = False

        self.feature_extractor = clip_model.vision_model
        self.foren_proj = nn.Linear(self.feature_dim, self.foren_dim, bias=False)
        self.fc = nn.Linear(self.foren_dim, int(num_classes), bias=False)

    def _vision_features(self, x):
        x = self.feature_extractor(x)["pooler_output"]
        x = self.foren_proj(x)
        return F.normalize(x, dim=1)


@MODELS.register_module(name=["GAPLStage1Model", "GAPLClipModel"])
class GAPLStage1Model(_GAPLBase):
    """Official GAPL stage-1 CLIP forensic projection model."""

    def __init__(self, **kwargs):
        model_conf = _as_plain_dict(kwargs.get("model", kwargs))
        super().__init__(**model_conf)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        full_state = super().state_dict(destination=None, prefix=prefix, keep_vars=keep_vars)
        out = {} if destination is None else destination
        target_prefix = prefix + "foren_proj."
        for key, value in full_state.items():
            if key.startswith(target_prefix):
                out[key] = value
        return out

    def forward(self, x):
        z = self._vision_features(x)
        logits = self.fc(z)
        return {"logits": logits, "z": z}


@MODELS.register_module(name=["GAPLModel", "GAPL", "gapl"])
class GAPLModel(nn.Module):
    """Generator-Aware Prototype Learning model, matched to the official implementation."""

    def __init__(self, **kwargs):
        super().__init__()
        model_conf = _as_plain_dict(kwargs.get("model", kwargs))

        self.feature_dim = int(model_conf.get("feature_dim", 1024))
        self.foren_dim = int(model_conf.get("foren_dim", 128))
        self.n_prototype = int(model_conf.get("n_prototype", 64))
        freeze_backbone = bool(model_conf.get("freeze_backbone", True))
        use_lora = bool(model_conf.get("use_lora", not freeze_backbone))

        clip_model = _load_clip_model(
            model_conf.get("clip_model_name_or_path", "openai/clip-vit-large-patch14"),
            local_files_only=bool(model_conf.get("local_files_only", False)),
            cache_dir=model_conf.get("cache_dir"),
        )

        if freeze_backbone:
            for param in clip_model.parameters():
                param.requires_grad = False

        if use_lora:
            try:
                from peft import LoraConfig, get_peft_model
            except ImportError as exc:
                raise ImportError(
                    "GAPL stage-2 LoRA fine-tuning requires `peft`. "
                    "Install project requirements again or run `pip install peft`."
                ) from exc

            lora_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=int(model_conf.get("lora_r", 16)),
                lora_alpha=int(model_conf.get("lora_alpha", 32)),
                lora_dropout=float(model_conf.get("lora_dropout", 0.1)),
                target_modules=list(model_conf.get("lora_target_modules", ["q_proj", "k_proj", "v_proj"])),
            )
            clip_model = get_peft_model(clip_model, lora_config)

        self.feature_extractor = clip_model.vision_model
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.foren_dim,
            num_heads=int(model_conf.get("num_heads", 4)),
            batch_first=True,
        )
        self.foren_proj = nn.Linear(self.feature_dim, self.foren_dim, bias=False)
        self.fc = nn.Linear(self.foren_dim, int(model_conf.get("num_classes", 1)), bias=False)

        self.register_buffer("proVec", torch.empty(0, self.foren_dim), persistent=False)

        fe_path = model_conf.get("fe_path")
        if fe_path:
            self.load_forensic_projection(fe_path)

        proto_path = model_conf.get("proto_path") or model_conf.get("prototype_path")
        if proto_path:
            self.load_prototype(torch.load(proto_path, map_location="cpu"))

    def load_forensic_projection(self, path):
        weight = _extract_foren_proj_weight(torch.load(path, map_location="cpu"))
        if tuple(weight.shape) != tuple(self.foren_proj.weight.shape):
            raise ValueError(
                f"GAPL forensic projection shape mismatch: expected "
                f"{tuple(self.foren_proj.weight.shape)}, got {tuple(weight.shape)}."
            )
        with torch.no_grad():
            self.foren_proj.weight.copy_(weight)

    def load_prototype(self, prototype):
        if isinstance(prototype, dict):
            for key in ("prototype", "proVec", "propVec"):
                if key in prototype:
                    prototype = prototype[key]
                    break
        if not torch.is_tensor(prototype):
            raise TypeError(f"Unsupported GAPL prototype type: {type(prototype)}")
        prototype = prototype.detach().to(dtype=self.foren_proj.weight.dtype)
        if prototype.ndim != 2 or prototype.shape[1] != self.foren_dim:
            raise ValueError(
                f"GAPL prototype must have shape [N, {self.foren_dim}], got {tuple(prototype.shape)}."
            )
        self.n_prototype = int(prototype.shape[0])
        self.proVec = prototype.to(self.proVec.device if self.proVec.numel() else self.foren_proj.weight.device)

    def forward(self, x, return_y=False):
        if self.proVec.numel() == 0:
            raise RuntimeError("GAPL prototypes are not loaded. Set model.proto_path or call load_prototype().")

        z = self.feature_extractor(x)["pooler_output"]
        z = self.foren_proj(z)
        z = F.normalize(z, dim=1)

        prototypes = self.proVec.to(device=z.device, dtype=z.dtype).unsqueeze(0).expand(z.shape[0], -1, -1)
        attended, weights = self.cross_attention(
            query=z.unsqueeze(1),
            key=prototypes,
            value=prototypes,
        )
        logits = self.fc(attended.squeeze(1))

        if return_y:
            return logits, weights
        return {"logits": logits, "z": z, "prototype_attention": weights}
