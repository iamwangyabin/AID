# 如何生成Arrow格式数据集

本项目支持使用高效的Arrow格式数据集。本文档介绍如何将原始标注和图像数据转换为Arrow格式。

---

## 1. 准备数据

### 图像文件
- 将所有图像文件放入一个根目录，例如：
```
dataset_root/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
├── class2/
│   ├── img3.jpg
│   └── img4.jpg
```
- 或者统一放在一个目录下，路径在标注中指定。

### 标注文件
- 使用JSON格式，支持单个文件或多个文件组成的目录。
- JSON结构示例：
```json
{
  "train": {
    "class1/img1.jpg": 0,
    "class2/img3.jpg": 1
  },
  "val": {
    "class1/img2.jpg": 0,
    "class2/img4.jpg": 1
  }
}
```
- 其中键为相对于`dataset_root`的图像路径，值为类别标签。

---

## 2. 生成Arrow数据集

在命令行中运行以下命令：

```bash
python tools/db_arrow_generator.py <json_dir> <dataset_root> <output_arrow_dir> <pool_num>
```

- `<json_dir>`：存放json标注文件的目录或单个json文件路径
- `<dataset_root>`：图像文件的根目录
- `<output_arrow_dir>`：生成的arrow数据集存放目录
- `<pool_num>`：多进程数量，遇到内存问题可设为1

### 示例

```bash
python tools/db_arrow_generator.py ./annotations ./images ./arrow_dataset 4
```

---

## 3. 生成结果

- 在`<output_arrow_dir>`目录下会生成：
  - Arrow格式的数据集
  - `mapping.json`：图片路径到索引的映射文件

---

## 4. 使用Arrow数据集

在训练或推理代码中，使用`data/arrow_datasets.py`中定义的类加载arrow数据集：

```python
from data.arrow_datasets import ArrowDatasets

dataset = ArrowDatasets(
    data_root="path/to/arrow_dataset",
    trsf=your_transform,
    subset="train",
    split="train"
)
```

- `data_root`：arrow数据集目录
- `subset`和`split`：对应json中的子集名称，如`train`、`val`
- `trsf`：数据增强或预处理transform

---

## 5. 注意事项

- 确保json中的路径是相对于`dataset_root`的相对路径。
- 多进程数量`pool_num`根据机器性能调整，内存不足时建议设为1。
- 支持多标签、多分类任务，具体格式可根据需求调整。

---

## 6. 参考

- `tools/db_arrow_generator.py`
- `data/arrow_datasets.py`