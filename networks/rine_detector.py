import torch
import torch.nn as nn
from torch.nn import functional as F

import clip
from utils.registry import MODELS

class Hook:
    def __init__(self, name, module):
        self.name = name
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output

    def close(self):
        self.hook.remove()


@MODELS.register_module()
class RINEModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        # Support nested 'model' dict or flat kwargs
        model_conf = kwargs.get('model', kwargs)

        # Compose 'backbone' tuple
        if 'backbone0' in model_conf and 'backbone1' in model_conf:
            backbone = (model_conf['backbone0'], model_conf['backbone1'])
        else:
            raise ValueError("RINEModel requires 'backbone0' and 'backbone1' in model config")

        # Extract 'nproj'
        if 'nproj' in model_conf:
            nproj = model_conf['nproj']
        else:
            raise ValueError("RINEModel requires 'nproj' parameter")

        # Extract 'proj_dim'
        if 'proj_dim' in model_conf:
            proj_dim = model_conf['proj_dim']
        else:
            raise ValueError("RINEModel requires 'proj_dim' parameter")

        # Load and freeze CLIP
        self.clip, self.preprocess = clip.load(backbone[0], device="cpu")
        for name, param in self.clip.named_parameters():
            param.requires_grad = False

        # Register hooks to get intermediate layer outputs
        self.hooks = [
            Hook(name, module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]

        # Initialize the trainable part of the model
        self.alpha = nn.Parameter(torch.randn([1, len(self.hooks), proj_dim]))
        proj1_layers = [nn.Dropout()]
        for i in range(nproj):
            proj1_layers.extend(
                [
                    nn.Linear(backbone[1] if i == 0 else proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*proj1_layers)
        proj2_layers = [nn.Dropout()]
        for _ in range(nproj):
            proj2_layers.extend(
                [
                    nn.Linear(proj_dim, proj_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj2 = nn.Sequential(*proj2_layers)
        self.head = nn.Sequential(
            *[
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, proj_dim),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(proj_dim, 1),
            ]
        )


    def forward(self, x):

        with torch.no_grad():
            self.clip.encode_image(x)
            g = torch.stack([h.output for h in self.hooks], dim=2)[0, :, :, :]

        g = self.proj1(g.float())

        z = torch.softmax(self.alpha, dim=1) * g

        z = torch.sum(z, dim=1)
        z = self.proj2(z)

        p = self.head(z)

        return {'logits': p, 'z': z}


