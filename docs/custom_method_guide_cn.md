# 如何在本仓库中实现和使用自定义方法

本指南介绍如何在本仓库中添加和使用你自己的模型或数据集类。

---

## 1. 实现自定义方法

你可以实现一个新的模型或数据集类。例如，定义一个新的模型：

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

仓库当前使用注册机制管理模型。相关注册表定义在`utils/registry.py`中：

- `MODELS`：模型

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

数据集类当前在 YAML 配置中通过 `data.ArrowDatasets` 这类 dotted name 引用。

---

## 3. 在配置文件中引用

在`cfgs/`目录下的yaml配置文件中，指定你注册的名称即可使用自定义方法。例如：

```yaml
arch: "MyCustomModel"
model:
  param1: value1
  param2: value2
```

同时需要在`utils/network_factory.py`的`MODEL_MODULE_MAP`中加入模型导入路径；导入后的模块必须在`MODELS`中注册同名`arch`。

---

## 4. 运行训练或推理

配置好yaml文件后，运行训练脚本：

```bash
python train.py --cfg cfgs/your_config.yaml
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
