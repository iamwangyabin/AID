# AID: AI-Generated Image Detection Toolbox & Benchmark

AID is a comprehensive toolbox and benchmark suite for detecting AI-generated (synthetic) images. It provides modular components for dataset handling, model training, evaluation, and benchmarking across various detection pipelines.

---

## Features

- Modular PyTorch + Lightning framework
- Supports multiple detection architectures (e.g., CLIP-based, CNNs, diffusion models)
- Configurable via YAML files with Hydra
- Extensive dataset support and augmentation
- Integrated experiment tracking with Weights & Biases (WandB)
- SLURM-compatible training/testing scripts
- Benchmarking on diverse datasets


Support methods:
注意当前仓库正在重构，目前经过验证能运行的是RINE，其余方法都需要debug




| Model Name  | Venue (Year) | Train | Test |
|:-----------:|:------------:|:-----:|:---:|
|   CNNDet    |   CVPR’20    |  ✔    |  ✔  |
|   GramNet   |   CVPR’20    |  ✔    |  ✔  |
|  FreDetect  |   ICML’20    |  ✘    |  ✔  |
|     LNP     |   ECCV’22    |  ✘    |  ✔  |
|   Fusing    |   ICIP’22    |  ✘    |  ✔  |
|    UniFD    |   CVPR’23    |  ✔    |  ✔  |
|    DIRE     |   ICCV’23    |  ✘    |  ✔  |
|    LGrad    |   CVPR’23    |  ✘    |  ✔  |
|  PoundNet   |   arXiv’24   |  ✔    |  ✔  |
|  CLIPping   |   ICMR’24    |  ✘    |  ✔  |
|  CLIPbased  |   CVPRW’24   |  ✘    |  ✔  |
|  FatFormer  |   CVPR’24    |  ✘    |     |
|   FreqNet   |   AAAI’24    |  ✔    |  ✔  |
|     NPR     |   CVPR’24    |  ✔    |  ✔  |
|    RINE     |   ECCV’24    |  ✔    |  ✔  |
|     DNF     |   ECAI’25    |  ✘    |  ✔  |
|    AIDE     |   ICLR’25    |  ✘    |  ✔  |
|    SPAI     |   CVPR’25    |  ✘    |  ✘   |







Support datasets:
在国内请用https://hf-mirror.com/ 代理后，执行类似方法进行数据集下载：
huggingface-cli download nebula/DF-arrow --repo-type dataset --local-dir /path/to/your/local/directory --local-dir-use-symlinks False







---

## Installation

1. **Clone the repository**

```bash
git clone 
cd AID
```

2. **Install dependencies**

It's recommended to use a virtual environment (e.g., conda or venv).

```bash
pip install -r requirements.txt
```

This will install packages including:

- OpenAI CLIP
- timm (PyTorch Image Models)
- Hydra
- WandB
- Lightning
- Albumentations
- scikit-learn
- OpenCV

---

## Usage

### Training

Prepare a training configuration YAML file (examples in `cfgs/train/`). Then run:

```bash
python train.py --cfg cfgs/train/your_config.yaml
```

This will:

- Load datasets and augmentations
- Initialize the specified model pipeline
- Log metrics to WandB
- Save checkpoints in `logs/`

**Example configs:**

- `cfgs/train/fakecoco/ojha_fcoco_sd3.yaml`
- `cfgs/train/fakepop/rine_fcoco_sd15.yaml`

You can also customize training via command-line overrides.

### Testing / Evaluation

Prepare a test configuration YAML file (examples in `cfgs/test/`). Then run:

```bash
python test.py --cfg cfgs/test/your_test_config.yaml
```

This will:

- Load the trained model checkpoint
- Evaluate on specified datasets
- Output metrics like AP, AUC, F1-score, accuracy

**Example configs:**

- `cfgs/test/t2ibenchmarks/rine_fakecoco_sd15.yaml`
- `cfgs/test/t2ibenchmarks/rine_fakepop_sd15.yaml`

### SLURM Batch Script

You can adapt `train.sh` for SLURM-based cluster training/testing. It contains example commands and environment setups.

---

## Project Structure

```
AID/
├── cfgs/             # YAML configs for training/testing
├── data/             # Dataset loaders and augmentations
├── engine/           # Training engine modules
├── networks/         # Model architectures
├── tools/            # Utility scripts and notebooks
├── train.py          # Training script
├── test.py           # Testing script
├── train.sh          # SLURM batch script example
├── requirements.txt  # Python dependencies
├── LICENSE           # License file
└── README.md         # This file
```

---

## Citation

If you use this toolbox in your research, please cite:

See `bibtex.md` for citation details.

---

## License

This project is licensed under the terms of the LICENSE file.

---

## Acknowledgments

- OpenAI CLIP
- PyTorch Lightning
- Hydra
- WandB
- And all contributors
