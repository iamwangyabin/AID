import torch
import timm
import torchvision

def resume_lightning(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # remove `model.` from key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)

def resume_timm(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict['backbone.'+key] = value
    model.load_state_dict(new_state_dict, False)


def resume_cnndet(model, weight_path):
    # used for resuming CNNDet original checkpoints
    state_dict = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])


def resume_rine(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    for name in state_dict:
        exec(
            f'model.{name.replace(".", "[", 1).replace(".", "].", 1)} = torch.nn.Parameter(state_dict["{name}"])'
        )
def resume_ojha(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    model.fc.load_state_dict(state_dict)


def no_resume(model, weight_path):
    """
    No-op resume function for models that don't need weight loading
    (e.g., zero-shot models that use pre-trained backbones)
    """
    pass

