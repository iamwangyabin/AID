import torch


def _load_checkpoint(weight_path, map_location='cpu'):
    try:
        return torch.load(weight_path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=map_location)
    except Exception:
        return torch.load(weight_path, map_location=map_location, weights_only=False)


def _get_nested_child(obj, name):
    if name.isdigit() and hasattr(obj, '__getitem__'):
        return obj[int(name)]
    return getattr(obj, name)


def _assign_nested_tensor(root, name, value):
    if not torch.is_tensor(value):
        raise TypeError(f'RINE checkpoint value for "{name}" must be a tensor, got {type(value).__name__}.')

    parent = root
    parts = name.split('.')
    for part in parts[:-1]:
        parent = _get_nested_child(parent, part)

    leaf = parts[-1]
    if isinstance(parent, torch.nn.Module) and leaf in parent._buffers:
        setattr(parent, leaf, value)
    else:
        setattr(parent, leaf, torch.nn.Parameter(value))


def resume_checkpoint(model, weight_path):
    checkpoint = _load_checkpoint(weight_path)
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
    state_dict = _load_checkpoint(weight_path)
    new_state_dict = {}
    for key, value in state_dict.items():
        new_state_dict['backbone.'+key] = value
    model.load_state_dict(new_state_dict, False)


def resume_cnndet(model, weight_path):
    # used for resuming CNNDet original checkpoints
    state_dict = _load_checkpoint(weight_path)
    model.load_state_dict(state_dict['model'])


def resume_rine(model, weight_path):
    state_dict = _load_checkpoint(weight_path)
    for name, value in state_dict.items():
        _assign_nested_tensor(model, name, value)


def resume_fire(model, weight_path):
    state_dict = _load_checkpoint(weight_path)
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
    state_dict = _load_checkpoint(weight_path)
    model.fc.load_state_dict(state_dict)


def resume_d3_attention_head(model, weight_path):
    """Load the official D3 classifier.pth attention-head checkpoint."""
    state_dict = _load_checkpoint(weight_path)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if any(key.startswith('attention_head.') for key in state_dict):
        state_dict = {key.replace('attention_head.', '', 1): value for key, value in state_dict.items()}
    model.attention_head.load_state_dict(state_dict)


def resume_vibnet(model, weight_path):
    checkpoint = _load_checkpoint(weight_path)
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
    checkpoint = _load_checkpoint(weight_path)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[7:]
        cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=True)


def resume_gapl(model, weight_path):
    checkpoint = _load_checkpoint(weight_path)

    if isinstance(checkpoint, dict) and 'prototype' in checkpoint:
        model.load_prototype(checkpoint['prototype'])
    elif isinstance(checkpoint, dict) and 'proVec' in checkpoint:
        model.load_prototype(checkpoint['proVec'])
    elif isinstance(checkpoint, dict) and 'propVec' in checkpoint:
        model.load_prototype(checkpoint['propVec'])

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        for prefix in ('module.', '_orig_mod.', 'model.'):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in {'prototype', 'proVec', 'propVec'}:
            model.load_prototype(value)
            continue
        if key.endswith('.proVec'):
            model.load_prototype(value)
            continue
        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"GAPL resume missing keys: {len(missing)}")
    if unexpected:
        print(f"GAPL resume unexpected keys: {len(unexpected)}")
