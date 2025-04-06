import torch
import torch.nn as nn

from clip import clip
from clip.model import convert_weights

from .coop import load_clip_to_cpu
from .imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT


class ZeroshotCLIP(nn.Module):
    def __init__(self, cfg, classnames, device):
        super().__init__()
        n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.model.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(device)
        for params in clip_model.parameters():
            params.requires_grad_(False)
        temp = "a photo of a {}."
        prompts = [temp.format(c.replace("_", " ")) for c in classnames]
        print(f"Prompts: {prompts}")
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(device)

        with torch.no_grad():
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self.text_features = text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits



class ZeroshotCLIP_PE(nn.Module):
    """Prompt ensembling."""

    templates = IMAGENET_TEMPLATES
    # templates = IMAGENET_TEMPLATES_SELECT


    def __init__(self, cfg, classnames, device):
        super().__init__()
        n_cls = len(classnames)

        print(f"Loading CLIP (backbone: {cfg.model.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model.to(device)

        for params in clip_model.parameters():
            params.requires_grad_(False)

        num_temp = len(self.templates)
        print(f"Prompt ensembling (n={num_temp})")

        mean_text_features = 0
        for i, temp in enumerate(self.templates):
            prompts = [temp.format(c.replace("_", " ")) for c in classnames]
            prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            text_features = clip_model.encode_text(prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            mean_text_features = mean_text_features + text_features
        mean_text_features = mean_text_features / num_temp
        mean_text_features = mean_text_features / mean_text_features.norm(dim=-1, keepdim=True)

        self.text_features = mean_text_features
        self.clip_model = clip_model

    def forward(self, image):
        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        return logits





