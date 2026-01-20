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


def resume_spai(model, weight_path):
    # PyTorch 2.6 defaults to weights_only=True; SPAI checkpoints may include non-tensor metadata.
    # Use weights_only=False to preserve prior behavior for trusted checkpoints.
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    if any(k.startswith('encoder.') for k in state_dict.keys()):
        state_dict = {
            k.replace('encoder.', ''): v
            for k, v in state_dict.items()
            if k.startswith('encoder.')
        }

    model.model.load_state_dict(state_dict, strict=False)
