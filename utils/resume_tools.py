import torch
import timm
import torchvision

def resume_checkpoint(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]  # remove `model.` from key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)


def resume_lightning(model, weight_path):
    resume_checkpoint(model, weight_path)

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


def resume_fire(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    state_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            key = key[6:]
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"FIRE resume missing keys: {len(missing)}")
    if unexpected:
        print(f"FIRE resume unexpected keys: {len(unexpected)}")


def resume_ojha(model, weight_path):
    state_dict = torch.load(weight_path, map_location='cpu')
    model.fc.load_state_dict(state_dict)


def resume_d3_attention_head(model, weight_path):
    """Load the official D3 classifier.pth attention-head checkpoint."""
    state_dict = torch.load(weight_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if any(key.startswith('attention_head.') for key in state_dict):
        state_dict = {key.replace('attention_head.', '', 1): value for key, value in state_dict.items()}
    model.attention_head.load_state_dict(state_dict)


def resume_vibnet(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    strip_trainer_prefix = False
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        strip_trainer_prefix = True
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        if strip_trainer_prefix and key.startswith('model.'):
            key = key[6:]
        cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=False)


def no_resume(model, weight_path):
    """
    No-op resume function for models that don't need weight loading
    (e.g., zero-shot models that use pre-trained backbones)
    """
    pass


def resume_bfree(model, weight_path):
    checkpoint = torch.load(weight_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=True)
