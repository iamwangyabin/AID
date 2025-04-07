import torch
import torchvision

from utils.registry import MODELS
def get_model(conf):
    print("Model loaded..")
    if hasattr(conf, 'arch') and conf.arch in MODELS:
        if hasattr(conf, 'model'):
            kwargs = conf.model
        else:
            kwargs = {}
        return MODELS.build(conf.arch, **kwargs)








