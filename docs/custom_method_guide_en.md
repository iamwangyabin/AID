# How to Implement and Use Custom Methods in This Repository

This guide explains how to add and use your own models, datasets, or post-processing functions within this repository. With the registry mechanism, you can easily extend the project's capabilities.

---

## 1. Implement Your Custom Method

You can implement a new model, dataset class, or post-processing function. For example, define a new model:

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Define network architecture
        ...

    def forward(self, x):
        # Define forward pass
        ...
```

---

## 2. Register Your Custom Method

The repository uses a registry system to manage models, datasets, and post-processing functions. The registries are defined in `utils/registry.py`:

- `MODELS`: for models
- `DATASETS`: for datasets
- `POSTFUNCS`: for post-processing functions

You can register your method using a decorator or a function call.

### Register with a Decorator

```python
from utils.registry import MODELS

@MODELS.register_module()
class MyCustomModel(nn.Module):
    ...
```

### Or Register with a Function Call

```python
from utils.registry import MODELS

MODELS.register_module(module=MyCustomModel)
```

For datasets and post-processing functions, use the corresponding `DATASETS` and `POSTFUNCS` registries.

---

## 3. Reference in Configuration Files

In the YAML configuration files under the `cfgs/` directory, specify the registered name to use your custom method. For example:

```yaml
model:
  name: MyCustomModel
  params:
    param1: value1
    param2: value2
```

The training script will automatically build the corresponding instance from the registry based on the configuration.

---

## 4. Run Training or Inference

After configuring the YAML file, run the training script:

```bash
python train.py --config cfgs/your_config.yaml
```

Or use the shell script:

```bash
bash train.sh
```

---

## 5. Notes

- Registry names are case-sensitive. Ensure the name in the config matches the registered name exactly.
- You can register multiple aliases for the same class. See `utils/registry.py` for details.
- If the name is misspelled, the system will suggest similar names.
- You can use `force=True` during registration to override existing entries with the same name.

---

## 6. References

- `utils/registry.py`
- Example configuration files in the `cfgs/` directory
- Trainer implementations in the `engine/` directory
- PyTorch official documentation