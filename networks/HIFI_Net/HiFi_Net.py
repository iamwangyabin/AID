import torch
import torch.nn as nn
import numpy as np
from networks.HIFI_Net.seg_hrnet import get_seg_model
from networks.HIFI_Net.seg_hrnet_config import get_cfg_defaults
from networks.HIFI_Net.NLCDetection_api import NLCDetection
from utils.registry import MODELS


@MODELS.register_module(name='HiFi_Net')
class HiFi_Net(nn.Module):
    def __init__(self):
        super(HiFi_Net, self).__init__()
        cfg = get_cfg_defaults()
        self.FENet = get_seg_model(cfg)
        weight_path = 'weights/HRNet/750001.pth'
        state_dict = torch.load(weight_path, map_location='cpu')['model']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.FENet.load_state_dict(new_state_dict)

        self.SegNet = NLCDetection()
        weight_path = 'weights/NLCDetection/750001.pth'
        state_dict = torch.load(weight_path, map_location='cpu')['model']
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.SegNet.load_state_dict(new_state_dict)

    def forward(self, x, **kwargs):
        output = self.FENet(x)
        mask1_fea, mask1_binary, out0, out1, out2, out3 = self.SegNet(output, x)
        x = nn.Softmax(dim=1)(out3)
        prob = 1 - x[:,0]
        return {'logits': prob,
                'features': None}

