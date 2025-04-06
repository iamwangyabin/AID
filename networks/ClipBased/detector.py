
import torch
import torch.nn as nn
from torch import load
from .openclipnet import OpenClipLinear
from .resnet_mod import resnet50


def load_weights(model, model_path):
    dat = load(model_path, map_location='cpu')
    if 'model' in dat:
        if ('module._conv_stem.weight' in dat['model']) or \
           ('module.fc.fc1.weight' in dat['model']) or \
           ('module.fc.weight' in dat['model']):
            model.load_state_dict(
                {key[7:]: dat['model'][key] for key in dat['model']})
        else:
            model.load_state_dict(dat['model'])
    elif 'state_dict' in dat:
        model.load_state_dict(dat['state_dict'])
    elif 'net' in dat:
        model.load_state_dict(dat['net'])
    elif 'main.0.weight' in dat:
        model.load_state_dict(dat)
    elif '_fc.weight' in dat:
        model.load_state_dict(dat)
    elif 'conv1.weight' in dat:
        model.load_state_dict(dat)
    else:
        print(list(dat.keys()))
        assert False
    return model


class CLIPBasedModel(nn.Module):
    def __init__(self, pretrained_path=None, num_classes=1):
        super(CLIPBasedModel, self).__init__()
        # res50nodown = resnet50(num_classes=1, stride0=1, dropout=0.5)

        self.backbone = OpenClipLinear(num_classes=num_classes, pretrain='clipL14commonpool', normalize=True,
                                               next_to_last=True)
        if pretrained_path is not None:
            load_weights(self.backbone, pretrained_path)
        self.backbone.to('cuda').eval()

    def forward(self, x, return_feature=False):
        out_tens = self.backbone(x)[:, 0]
        return out_tens







