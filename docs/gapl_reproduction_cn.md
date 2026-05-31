# GAPL 官方复现接入说明

官方代码与权重来源：

- Code: https://github.com/UltraCapture/GAPL
- Checkpoints: https://huggingface.co/AbyssLumine/GAPL
- Stage-2 data: https://huggingface.co/datasets/OwensLab/CommunityForensics-Small

## 准备权重

```bash
pip install -r requirements.txt
python tools/download_gapl_weights.py
```

下载脚本会把 Hugging Face 上的 `checkpoint.pt`、`stage1.pt` 和 prototype 文件整理到：

```text
networks/weights/gapl/
```

## 直接测试官方 checkpoint

先修改 `cfgs/test/official/gapl.yaml` 中的 benchmark 数据路径，然后运行：

```bash
python test.py --cfg cfgs/test/official/gapl.yaml
```

也可以用命令行覆盖路径：

```bash
python test.py --cfg cfgs/test/official/gapl.yaml datasets.base_path=/your/DF-arrow/path
```

## Stage 1 训练

Stage 1 对应官方 `ClipModel`：冻结 CLIP ViT-L/14，仅学习 128 维 forensic projection。

```bash
python train.py --cfg cfgs/train/official/gapl_stage1.yaml datasets.base_path=/your/CNNDetection/path
```

官方 stage1 checkpoint 是 `foren_proj.weight` tensor；本仓库的 `GAPLStage1Model.state_dict()` 也只导出该投影层，方便后续 prototype 提取或 stage2 初始化。

## Stage 2 训练

Stage 2 对应官方 `GAPLModel`：加载 stage1 projection 和 generator-aware prototypes，在 CLIP q/k/v 上加 LoRA，并通过 prototype cross-attention 做二分类。

```bash
python train.py --cfg cfgs/train/official/gapl_stage2.yaml
```

如需使用离线 CLIP 权重，将 `model.clip_model_name_or_path` 改成本地 `openai/clip-vit-large-patch14` snapshot 路径，并设置：

```yaml
model:
  local_files_only: true
```
