import importlib
from typing import Any, Optional

from utils.registry import MODELS

# 模型到模块的映射表
MODEL_MODULE_MAP = {
    'NPRModel': 'networks.npr_detector',
    'FreqNet': 'networks.freqnet_detector', 
    'GramNet': 'networks.gramnet_detector',
    'CNNDet': 'networks.cnndet',
    'LOTAModel': 'networks.lota_detector',
    'LOTA': 'networks.lota_detector',
    
    'RINEModel': 'networks.rine_detector',
    'OjhaModel': 'networks.ojha_detector',
    'PoundNet': 'networks.poundnet_detector',
    
    'ManifoldInducedBiases': 'networks.sdv14_detector',
    'RIGIDModel': 'networks.rigid_detector',
    'WaRPADModel': 'networks.warpad_detector',
    'CLIDEModel': 'networks.clide_detector',
    'CLIDE': 'networks.clide_detector',
    'FIREModel': 'networks.fire_detector',
    'FIRE': 'networks.fire_detector',
    
    'HiFi_Net': 'networks.HIFI_Net.HiFi_Net',
    'TIMMModel': 'networks.timm_detector',
    'VJEPA2_1Linear': 'networks.vjepa2_1_detector',
    'BFreeModel': 'networks.bfree_detector',
    'D3Model': 'networks.d3_detector',
    'VIBNet': 'networks.vibnet_detector',
    'SPAIModel': 'networks.spai_detector',
    'SPAIMFM': 'networks.spai_detector',
    'GAPLStage1Model': 'networks.gapl_detector',
    'GAPLClipModel': 'networks.gapl_detector',
    'GAPLModel': 'networks.gapl_detector',
    'GAPL': 'networks.gapl_detector',
    'gapl': 'networks.gapl_detector',

}

def load_model_module(model_name: str) -> Optional[Any]:
    module_path = MODEL_MODULE_MAP.get(model_name)
    if module_path is None:
        available = ", ".join(sorted(MODEL_MODULE_MAP))
        raise KeyError(f'Unknown model arch "{model_name}". Available arch names: {available}')
    module = importlib.import_module(module_path)
    return module

def get_model(conf):
    print("Loading model...")
    model_name = conf.arch
    load_model_module(model_name)

    if model_name in MODELS:
        if model_name == "PoundNet":
            return MODELS.build(model_name, conf)
        if hasattr(conf, 'model'):
            kwargs = conf.model
        else:
            kwargs = {}
        return MODELS.build(model_name, **kwargs)
    raise KeyError(f'"{model_name}" was imported but did not register itself in MODELS.')
