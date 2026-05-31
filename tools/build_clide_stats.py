#!/usr/bin/env python3
"""
Build CLIDE representative embeddings and global whitening stats from real images.

This mirrors the official CLIDE detection.py preprocessing path:
PIL image -> OpenAI CLIP preprocess -> CLIP image embedding -> optional PCA whitening.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import clip
except ImportError:
    from networks.SPrompts.clip import clip


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def list_image_files(directory: str | Path) -> List[Path]:
    directory = Path(directory)
    files = [
        directory / filename
        for filename in os.listdir(directory)
        if filename.lower().endswith(IMAGE_EXTS)
    ]
    files.sort()
    return files


def sphx(x: torch.Tensor, m: int = 500, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D tensor, got {x.ndim}D.")
    n, d = x.shape
    if n < 2:
        raise ValueError("At least two embeddings are required for whitening.")

    m = min(int(m), d, max(1, n - 1))
    x = x.float()
    xu = x - x.mean(dim=0)
    cov_matrix = torch.cov(xu.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
    indices = torch.argsort(eigenvalues, descending=True)[:m]
    rotation_matrix = eigenvectors[:, indices] @ torch.diag(torch.rsqrt(eigenvalues[indices].clamp_min(eps)))
    return xu @ rotation_matrix, rotation_matrix


@torch.no_grad()
def create_rep_set(
    image_dir: str | Path,
    output_path: str | Path,
    clip_name: str = "ViT-L/14",
    batch_size: int = 64,
    device: str | None = None,
) -> torch.Tensor:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(clip_name, device=device)
    model.eval()

    image_files = list_image_files(image_dir)
    if not image_files:
        raise ValueError(f"No images found in {image_dir}.")

    embeddings = []
    for start in tqdm(range(0, len(image_files), batch_size), desc="Embedding CLIDE rep set"):
        batch_files = image_files[start : start + batch_size]
        batch = []
        for image_file in batch_files:
            image = Image.open(image_file).convert("RGB")
            batch.append(preprocess(image))
        image_tensor = torch.stack(batch, dim=0).to(device)
        emb = model.encode_image(image_tensor).float().cpu()
        embeddings.append(emb)

    rep_embeddings = torch.cat(embeddings, dim=0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(rep_embeddings, output_path)
    print(f"saved rep matrix {tuple(rep_embeddings.shape)} -> {output_path}")
    return rep_embeddings


def create_global_w_mat(rep_embeddings: torch.Tensor, output_path: str | Path, m: int | None = None) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rep_embeddings = rep_embeddings.to(device)
    _, w_mat = sphx(rep_embeddings, m=m or rep_embeddings.shape[1])
    w_mean = rep_embeddings.mean(dim=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((w_mat.cpu(), w_mean.cpu()), output_path)
    print(f"saved whitening matrix {tuple(w_mat.shape)} -> {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rep-dir", required=True, help="Directory of real representative images.")
    parser.add_argument("--rep-mat-path", default="networks/weights/clide/rep_matrix_custom.pt")
    parser.add_argument("--w-mat-path", default="networks/weights/clide/whitening_matrix_custom.pt")
    parser.add_argument("--clip-name", default="ViT-L/14")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--m", type=int, default=None, help="Global whitening dimensions. Defaults to CLIP dimension.")
    parser.add_argument("--skip-rep", action="store_true", help="Reuse an existing --rep-mat-path.")
    parser.add_argument("--skip-global", action="store_true", help="Only create the representative matrix.")
    args = parser.parse_args()

    if args.skip_rep:
        rep_embeddings = torch.load(args.rep_mat_path, map_location="cpu")
    else:
        rep_embeddings = create_rep_set(args.rep_dir, args.rep_mat_path, args.clip_name, args.batch_size)

    if not args.skip_global:
        create_global_w_mat(rep_embeddings, args.w_mat_path, args.m)


if __name__ == "__main__":
    main()
