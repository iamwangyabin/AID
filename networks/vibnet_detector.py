import contextlib

import clip
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.registry import MODELS


CHANNELS = {
    "RN50": 1024,
    "ViT-L/14": 768,
    "RN50x64": 1024,
    "ViT-L/14@336px": 768,
}


@MODELS.register_module()
class VIBNet(nn.Module):
    """CLIP feature detector with a variational information bottleneck."""

    def __init__(
        self,
        name="ViT-L/14",
        bottleneck_dim=256,
        hidden_dim=1024,
        dropout=0.5,
        num_classes=1,
        freeze_clip=True,
        sample_latent=True,
        use_mean_on_eval=False,
        feature_dim=None,
    ):
        super().__init__()
        self.k = bottleneck_dim
        self.freeze_clip = freeze_clip
        self.sample_latent = sample_latent
        self.use_mean_on_eval = use_mean_on_eval

        self.model, self.preprocess = clip.load(name, device="cpu")
        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False

        in_dim = feature_dim or CHANNELS[name]
        self.fc_1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU(True)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 2 * self.k)
        self.decode = nn.Linear(self.k, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, **kwargs):
        no_grad = torch.no_grad() if self.freeze_clip else contextlib.nullcontext()
        with no_grad:
            features = self.model.encode_image(x)

        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = features.float()

        statistics = self.dropout(features)
        statistics = self.relu(self.fc_1(statistics))
        statistics = self.relu(self.fc_2(statistics))
        statistics = self.fc_3(statistics)

        mu = statistics[:, : self.k]
        std = F.softplus(statistics[:, self.k :] - 5, beta=1)
        z = self.reparameterize(mu, std)
        logits = self.decode(z)
        kl = self.kl_divergence(mu, std)

        return {
            "logits": logits,
            "features": features,
            "z": z,
            "mu": mu,
            "std": std,
            "kl": kl,
            "distribution": (mu, std),
        }

    def reparameterize(self, mu, std):
        if (not self.training and self.use_mean_on_eval) or not self.sample_latent:
            return mu
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu, std):
        var = std.pow(2)
        kl_per_sample = 0.5 * torch.sum(mu.pow(2) + var - torch.log(var + 1e-8) - 1, dim=1)
        return kl_per_sample.mean()
