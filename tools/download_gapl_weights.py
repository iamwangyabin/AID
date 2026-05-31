import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


CANONICAL_FILES = {
    "checkpoint.pt": ("checkpoint.pt",),
    "stage1.pt": ("stage1.pt", "pretrained/stage1.pt"),
    "pca_3type_num64.pt": ("pca_3type_num64.pt", "prototype/pca_3type_num64.pt"),
    "pca_3type_num32.pt": ("pca_3type_num32.pt", "prototype/pca_3type_num32.pt"),
    "pca_3type_num16.pt": ("pca_3type_num16.pt", "prototype/pca_3type_num16.pt"),
}


def canonicalize_downloads(local_dir):
    local_dir = Path(local_dir)
    for output_name, candidates in CANONICAL_FILES.items():
        output_path = local_dir / output_name
        if output_path.exists():
            continue
        for candidate in candidates:
            candidate_path = local_dir / candidate
            if candidate_path.exists():
                shutil.copy2(candidate_path, output_path)
                break


def main():
    parser = argparse.ArgumentParser(description="Download official GAPL checkpoints from Hugging Face.")
    parser.add_argument("--repo-id", default="AbyssLumine/GAPL")
    parser.add_argument("--local-dir", default="networks/weights/gapl")
    args = parser.parse_args()

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        allow_patterns=["*.pt", "pretrained/*.pt", "prototype/*.pt"],
        local_dir_use_symlinks=False,
    )
    canonicalize_downloads(args.local_dir)
    print(f"GAPL weights downloaded to {Path(args.local_dir).resolve()}")


if __name__ == "__main__":
    main()
