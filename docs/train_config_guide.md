# Training Configuration File Guide

This document explains how to write training configuration files for this project, including the typical structure and configurable options.

## Overview

Training configuration files are written in YAML format. They define the model architecture, datasets, training/testing parameters, and data processing pipelines. These configs enable reproducible experiments and flexible customization.

## Typical Structure

A configuration file usually contains the following sections:

### 1. Metadata

- `arch`: The registered model architecture name (e.g., `"RINEModel"`, `"NPRModel"`)
- `test_name`: A descriptive name for the experiment or test
- Comments can be added using `#` to describe the config

### 2. Evaluation Pipeline

- `eval_pipeline`: The function or pipeline used for evaluation (e.g., `utils.validate_plain`)

### 3. Resume Checkpoint

- `resume.path`: Path to a pretrained model or checkpoint file
- `resume.target`: Function to handle loading the checkpoint

### 4. Model Parameters

Defines the model architecture and hyperparameters. Example:

```yaml
model:
  backbone0: "ViT-L/14"     # Backbone model name
  backbone1: 1024           # Additional backbone parameter
  nproj: 2                  # Number of projection layers
  proj_dim: 1024            # Projection dimension
```

### 5. Datasets

Defines datasets used for training or testing.

- `base_path`: Base directory for datasets
- `source`: List of datasets with details:

```yaml
datasets:
  base_path: "/path/to/datasets"
  source:
    - target: data.ArrowDatasets
      data_root: "${datasets.base_path}/ForenSynths"
      sub_sets: ["biggan", "crn", "cyclegan", "deepfake"]
      split: "test"
      benchmark_name: "ForenSynths"

    - target: data.BinaryJsonDatasets
      data_root: "/path/to/another/dataset"
      sub_sets: ["dalle2", "firefly"]
      split: "test"
      benchmark_name: "Synthbuster"
```

You can specify multiple datasets, each with:

- `target`: Dataset class/type
- `data_root`: Path to dataset
- `sub_sets`: List of categories or subsets
- `split`: Data split (e.g., `"train"`, `"test"`)
- `benchmark_name`: Name of the dataset group

### 6. DataLoader Settings

- `batch_size`: Number of samples per batch
- `loader_workers`: Number of worker threads for data loading

### 7. Data Transformations and Augmentations

Data transformations (`trsf`) define the preprocessing and augmentation pipeline applied to images before feeding them into the model.

#### Purpose of Transformations

- **Preprocessing:** Resize, crop, normalize images to a standard format compatible with the model.
- **Augmentation:** Apply random changes to increase data diversity, improving model robustness and generalization.

#### Example Transformation Pipeline

```yaml
trsf:
  - _target_: data.RandomCompress
    method: "JPEG"
    qf: [70, 100]

  - _target_: torchvision.transforms.Resize
    size: 256

  - _target_: torchvision.transforms.CenterCrop
    size: 224

  - _target_: torchvision.transforms.ToTensor

  - _target_: torchvision.transforms.Normalize
    mean: [0.48145466, 0.4578275, 0.40821073]
    std: [0.26862954, 0.26130258, 0.27577711]
```

#### Common Transformations

- **Compression Augmentation**
  - `data.RandomCompress`: Simulates compression artifacts (e.g., JPEG) to improve robustness.
  - Parameters:
    - `method`: Compression type (e.g., `"JPEG"`)
    - `qf`: Quality factor range (lower = more compression)

- **Geometric Transforms**
  - `torchvision.transforms.Resize`: Resize image to a fixed size.
  - `torchvision.transforms.CenterCrop`: Crop the center region.
  - `torchvision.transforms.RandomResizedCrop`: Random crop and resize (useful for training).
  - `torchvision.transforms.RandomHorizontalFlip`: Randomly flip images horizontally.
  - `torchvision.transforms.RandomRotation`: Randomly rotate images within a degree range.

- **Color and Intensity Augmentations**
  - `torchvision.transforms.ColorJitter`: Randomly change brightness, contrast, saturation, hue.
  - `torchvision.transforms.RandomGrayscale`: Convert images to grayscale with a probability.
  - `torchvision.transforms.GaussianBlur`: Apply blur to simulate out-of-focus images.

- **Conversion and Normalization**
  - `torchvision.transforms.ToTensor`: Convert PIL image to tensor.
  - `torchvision.transforms.Normalize`: Normalize tensor with mean and std.

#### Tips for Using Augmentations

- **Training vs. Evaluation:**
  - Use **random augmentations** (flip, crop, jitter) during training to improve generalization.
  - Use **deterministic transforms** (resize, center crop, normalize) during validation/testing for consistency.

- **Order Matters:**
  - Apply geometric and color augmentations **before** converting to tensor.
  - Normalize **after** converting to tensor.

- **Custom Augmentations:**
  - You can implement custom transforms (e.g., `data.RandomCompress`) and include them in the pipeline.

- **Adjusting Augmentations:**
  - Tune augmentation strength based on dataset characteristics.
  - Excessive augmentation may harm performance if unrealistic.

---

## Additional Tips

- Use `${variable}` syntax to reference other config values.
- Comment out alternative datasets or settings for easy switching.
- Organize configs by experiment purpose (e.g., training, testing, ablation).
- Store configs under `cfgs/` directory for better management.

## Example Snippet

```yaml
arch: "RINEModel"
test_name: "RINE_official_resize"

eval_pipeline: utils.validate_plain

resume:
  path: './networks/weights/rine_model_4class_trainable.pth'
  target: utils.resume_tools.resume_rine

model:
  backbone0: "ViT-L/14"
  backbone1: 1024
  nproj: 2
  proj_dim: 1024

datasets:
  base_path: "/home/user/data"
  source:
    - target: data.ArrowDatasets
      data_root: "${datasets.base_path}/ForenSynths"
      sub_sets: ["biggan", "crn", "cyclegan", "deepfake"]
      split: "test"
      benchmark_name: "ForenSynths"

batch_size: 64
loader_workers: 32

trsf:
  - _target_: data.RandomCompress
    method: "JPEG"
    qf: [70, 100]
  - _target_: torchvision.transforms.Resize
    size: 256
  - _target_: torchvision.transforms.CenterCrop
    size: 224
  - _target_: torchvision.transforms.ToTensor
  - _target_: torchvision.transforms.Normalize
    mean: [0.48145466, 0.4578275, 0.40821073]
    std: [0.26862954, 0.26130258, 0.27577711]
```

---

## Summary

The training configuration file allows you to customize:

- Model architecture and parameters
- Datasets and data splits
- Checkpoints and resume options
- Data loading settings
- **Data augmentation and preprocessing pipeline**

Carefully design your augmentation pipeline to balance data diversity and realism, improving model robustness and generalization.
