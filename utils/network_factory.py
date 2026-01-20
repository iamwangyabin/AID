import torch
import torchvision
import importlib
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from utils.registry import MODELS

# 模型到模块的映射表
MODEL_MODULE_MAP = {
    'NPRModel': 'networks.npr_detector',
    'FreqNet': 'networks.freqnet_detector', 
    'GramNet': 'networks.gramnet_detector',
    'CNNDet': 'networks.cnndet',
    
    'RINEModel': 'networks.rine_detector',
    'OjhaModel': 'networks.ojha_detector',
    'PoundNet': 'networks.poundnet_detector',
    
    'ManifoldInducedBiases': 'networks.sdv14_detector',
    'RIGIDModel': 'networks.rigid_detector',
    'WaRPADModel': 'networks.warpad_detector',
    
    'HiFi_Net': 'networks.HIFI_Net.HiFi_Net',
    'TIMMModel': 'networks.timm_detector',
    'SPAIModel': 'networks.spai_detector',

}

def load_model_module(model_name: str) -> Optional[Any]:
    module_path = MODEL_MODULE_MAP.get(model_name)
    module = importlib.import_module(module_path)
    return module

def get_model(conf):
    print("Loading model...")
    model_name = conf.arch
    load_model_module(model_name)

    if model_name in MODELS:
        if hasattr(conf, 'model'):
            kwargs = conf.model
        else:
            kwargs = {}
        return MODELS.build(model_name, **kwargs)
