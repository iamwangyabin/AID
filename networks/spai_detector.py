from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.registry import MODELS
from networks.spai.sid import build_mf_vit


def _deep_update(base, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [ _to_namespace(v) for v in obj ]
    return obj


def _default_spai_config():
    return {
        "DATA": {
            "IMG_SIZE": 224,
        },
        "MODEL": {
            "TYPE": "vit",
            "NUM_CLASSES": 2,
            "DROP_RATE": 0.0,
            "DROP_PATH_RATE": 0.1,
            "SID_DROPOUT": 0.5,
            "RESOLUTION_MODE": "arbitrary",
            "FEATURE_EXTRACTION_BATCH": 400,
            "VIT": {
                "PATCH_SIZE": 16,
                "IN_CHANS": 3,
                "EMBED_DIM": 768,
                "DEPTH": 12,
                "NUM_HEADS": 12,
                "MLP_RATIO": 4,
                "QKV_BIAS": True,
                "INIT_VALUES": None,
                "USE_APE": True,
                "USE_RPB": False,
                "USE_SHARED_RPB": False,
                "USE_MEAN_POOLING": True,
                "USE_INTERMEDIATE_LAYERS": True,
                "INTERMEDIATE_LAYERS": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                "PROJECTION_DIM": 1024,
                "PROJECTION_LAYERS": 2,
                "PATCH_PROJECTION": True,
                "PATCH_PROJECTION_PER_FEATURE": True,
            },
            "FRE": {
                "MASKING_RADIUS": 16,
                "PROJECTOR_LAST_LAYER_ACTIVATION_TYPE": None,
                "ORIGINAL_IMAGE_FEATURES_BRANCH": True,
                "DISABLE_RECONSTRUCTION_SIMILARITY": False,
            },
            "PATCH_VIT": {
                "PATCH_STRIDE": 224,
                "NUM_HEADS": 12,
                "ATTN_EMBED_DIM": 1536,
                "MINIMUM_PATCHES": 4,
            },
            "CLS_HEAD": {
                "MLP_RATIO": 3,
            },
        },
    }


@MODELS.register_module()
class SPAIModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_conf = kwargs.get("model", kwargs)

        config = _default_spai_config()
        overrides = model_conf.get("config_overrides", {})
        if overrides:
            _deep_update(config, overrides)

        self.spai_config = _to_namespace(config)
        self.model = build_mf_vit(self.spai_config)
        self.feature_extraction_batch_size = self.spai_config.MODEL.FEATURE_EXTRACTION_BATCH

    def forward(self, x: torch.Tensor) -> dict:
        if self.spai_config.MODEL.RESOLUTION_MODE == "arbitrary":
            logits = self.model(x, feature_extraction_batch_size=self.feature_extraction_batch_size)
        else:
            logits = self.model(x)
        return {"logits": logits}
