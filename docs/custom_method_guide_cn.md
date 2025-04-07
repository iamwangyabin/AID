# 如何在本仓库中实现和使用自定义方法

本指南介绍如何在本仓库中添加和使用你自己的模型、数据集或后处理方法。通过注册机制，你可以方便地扩展本项目的功能。

---

## 1. 实现自定义方法

你可以实现一个新的模型、数据集类或后处理函数。例如，定义一个新的模型：

```python
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # 定义网络结构
        ...

    def forward(self, x):
        # 定义前向传播
        ...
```

---

## 2. 注册自定义方法

仓库使用注册机制管理模型、数据集和后处理函数。相关注册表定义在`utils/registry.py`中：

- `MODELS`：模型
- `DATASETS`：数据集
- `POSTFUNCS`：后处理函数

你可以通过装饰器或函数调用注册你的方法。

### 使用装饰器注册

```python
from utils.registry import MODELS

@MODELS.register_module()
class MyCustomModel(nn.Module):
    ...
```

### 或者使用函数调用注册

```python
from utils.registry import MODELS

MODELS.register_module(module=MyCustomModel)
```

对于数据集和后处理函数，使用对应的`DATASETS`和`POSTFUNCS`注册表。

---

## 3. 在配置文件中引用

在`cfgs/`目录下的yaml配置文件中，指定你注册的名称即可使用自定义方法。例如：

```yaml
model:
  name: MyCustomModel
  params:
    param1: value1
    param2: value2
```

训练脚本会根据配置自动从注册表中构建对应的实例。

---

## 4. 运行训练或推理

配置好yaml文件后，运行训练脚本：

```bash
python train.py --config cfgs/your_config.yaml
```

或使用shell脚本：

```bash
bash train.sh
```

---

## 5. 注意事项

- 注册名称区分大小写，确保配置文件中的名称与注册时一致。
- 可以为同一个类注册多个别名，详见`utils/registry.py`。
- 如果名称拼写错误，系统会提示相似名称建议。
- 注册时可使用`force=True`覆盖已有同名项。

---

## 6. 参考

- `utils/registry.py`
- `cfgs/`目录下的示例配置文件
- `engine/`目录下的训练器实现
- PyTorch官方文档