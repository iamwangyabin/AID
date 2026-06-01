# AID: AI-Generated Image Detection Toolbox & Benchmark

AID is a comprehensive toolbox and benchmark suite for detecting AI-generated (synthetic) images. It provides modular components for dataset handling, model training, evaluation, and benchmarking across various detection pipelines.

---

## Features

- Modular PyTorch training and evaluation framework
- Supports multiple detection architectures (e.g., CLIP-based, CNNs, diffusion models)
- Configurable via YAML files with Hydra
- Extensive dataset support and augmentation
- Integrated experiment tracking with Weights & Biases (WandB)
- Shell templates for training/testing entry points
- Benchmarking on diverse datasets


Support methods:
注意当前仓库正在重构，目前经过验证能运行的是 RINE，其余方法都需要 debug。

| Method | Publication | Train | Test |
|:--|:--:|:--:|:--:|
| [CNNDet](https://github.com/peterwang512/CNNDetection) | CVPR 2020 | ✔ | ✔ |
| [GramNet](https://github.com/liuzhengzhe/Global_Texture_Enhancement_for_Fake_Face_Detection_in_the-Wild) | CVPR 2020 | ✔ | ✔ |
| [FreDetect](https://github.com/RUB-SysSec/GANDCTAnalysis) | ICML 2020 | ✘ | ✔ |
| [LNP](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136740089.pdf) | ECCV 2022 | ✘ | ✔ |
| [Fusing](https://github.com/littlejuyan/FusingGlobalandLocal) | ICIP 2022 | ✘ | ✔ |
| [UniFD](https://github.com/Yuheng-Li/UniversalFakeDetect) | CVPR 2023 | ✔ | ✔ |
| [DIRE](https://github.com/ZhendongWang6/DIRE) | ICCV 2023 | ✘ | ✔ |
| [LGrad](https://github.com/chuangchuangtan/LGrad) | CVPR 2023 | ✘ | ✔ |
| [DNF](https://github.com/YichiCS/Diffusion-Noise-Feature) | arXiv 2023 | ✘ | ✔ |
| [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection) | CVPR 2024 | ✔ | ✔ |
| [FreqNet](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection) | AAAI 2024 | ✔ | ✔ |
| [RINE](https://github.com/mever-team/rine) | ECCV 2024 | ✔ | ✔ |
| [FatFormer](https://github.com/Michel-liu/FatFormer) | CVPR 2024 | ✘ | ✘ |
| [CLIPping](https://github.com/sfimediafutures/CLIPping-the-Deception) | ICMR 2024 | ✘ | ✔ |
| [CLIPbased](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) | CVPRW 2024 | ✘ | ✔ |
| [AIDE](https://github.com/shilinyan99/AIDE) | ICLR 2025 | ✘ | ✔ |
| [FIRE](https://github.com/Chuchad/FIRE) | CVPR 2025 | ✔ | ✔ |
| [B-Free](https://github.com/grip-unina/B-Free) | CVPR 2025 | ✔ | ✔ |
| [GAPL](https://github.com/UltraCapture/GAPL) | CVPR 2026 | ✔ | ✔ |
| [PoundNet](https://github.com/iamwangyabin/PoundNet) | TPAMI 2026 | ✔ | ✔ |
| [CLIDE](https://github.com/FujitsuResearch/domain-adaptive-image-detection) | WACV 2026 | ✘ | ✔ |


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
- PyTorch
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

- `cfgs/train/official/rine_official.yaml`
- `cfgs/train/official/npr.yaml`

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

- `cfgs/test/official/rine_official.yaml`
- `cfgs/test/official/npr.yaml`

### Shell Scripts

You can adapt `train.sh` and `test.sh` for local or cluster runs. Set `AID_TRAIN_CFG` or `AID_TEST_CFG` to override the default config path.

---

## Project Structure

```
AID/
├── cfgs/             # YAML configs for training/testing
├── data/             # Dataset loaders and augmentations
├── engine/           # Training engine modules
├── networks/         # Model architectures
├── tools/            # Utility scripts
├── train.py          # Training script
├── test.py           # Testing script
├── train.sh          # training shell template
├── requirements.txt  # Python dependencies
├── LICENSE           # License file
└── README.md         # This file
```

---

## Citation

If you use this toolbox in your research, please cite:

See `docs/bibtex.md` for citation details.

---

## License

This project is licensed under the terms of the LICENSE file.

---

## Acknowledgments

- OpenAI CLIP
- PyTorch
- Hydra
- WandB
- And all contributors
