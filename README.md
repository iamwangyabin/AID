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

|    Model Name     | Train | Test |
|:-----------------:|:----:|:----:|
|      CNNDet       |   ✔  |  ✔   |
|      GramNet      |   ✔  |  ✔   |
|      FreqNet      |   ✔  |  ✔   |
|        NPR        |   ✔  |  ✔   |
|       RINE        |   ✔  |  ✔   |
|       UniFD       |   ✔  |  ✔   |
|       DIRE        |   ✘  |  ✔   |
|     DNF      |   ✘    |   ✔    |
|       LGrad       |   ✘  |  ✔   |
|        LNP        |   ✘  |  ✔   |
|      Fusing       |   ✘  |  ✔   |
|     PoundNet      |  ✔   |  ✔   |
|     CLIPping      |  ✘    |  ✔   |
|     CLIPbased     |   ✘   |  ✔   |
|     FatFormer     |    ✘  |      |
|     FreDetect     |   ✘   |  ✔   |
|     AIDE     |   ✘   |  ✔   |





Support datasets:




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
