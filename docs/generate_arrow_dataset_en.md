# How to Generate Arrow Format Dataset

This project supports efficient datasets in Arrow format. This document explains how to convert your original annotations and image data into Arrow format.

---

## 1. Prepare Your Data

### Image Files
- Place all image files under a root directory, for example:
```
dataset_root/
├── class1/
│   ├── img1.jpg
│   └── img2.jpg
├── class2/
│   ├── img3.jpg
│   └── img4.jpg
```
- Or put all images in a single directory, with paths specified in annotations.

### Annotation Files
- Use JSON format, either a single file or a directory containing multiple JSON files.
- Example JSON structure:
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
- Keys are image paths relative to `dataset_root`, values are class labels.

---

## 2. Generate Arrow Dataset

Run the following command in your terminal:

```bash
python tools/db_arrow_generator.py <json_dir> <dataset_root> <output_arrow_dir> <pool_num>
```

- `<json_dir>`: Directory containing JSON annotation files or a single JSON file path
- `<dataset_root>`: Root directory of image files
- `<output_arrow_dir>`: Directory to save the generated Arrow dataset
- `<pool_num>`: Number of processes for multiprocessing, set to 1 if you encounter memory issues

### Example

```bash
python tools/db_arrow_generator.py ./annotations ./images ./arrow_dataset 4
```

---

## 3. Output

- The `<output_arrow_dir>` will contain:
  - The Arrow format dataset
  - `mapping.json`: a mapping file from image paths to indices

---

## 4. Using the Arrow Dataset

In your training or inference code, use the classes defined in `data/arrow_datasets.py` to load the Arrow dataset:

```python
from data.arrow_datasets import ArrowDatasets

dataset = ArrowDatasets(
    data_root="path/to/arrow_dataset",
    trsf=your_transform,
    subset="train",
    split="train"
)
```

- `data_root`: directory of the Arrow dataset
- `subset` and `split`: subset names in your JSON, e.g., `train`, `val`
- `trsf`: your data augmentation or preprocessing transform

---

## 5. Notes

- Ensure paths in JSON are relative to `dataset_root`.
- Adjust `pool_num` based on your hardware; set to 1 if memory is limited.
- Supports multi-label and multi-class tasks; adjust JSON format accordingly.

---

## 6. References

- `tools/db_arrow_generator.py`
- `data/arrow_datasets.py`