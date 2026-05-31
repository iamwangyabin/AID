#!/usr/bin/env python3
"""Download official CLIDE precomputed whitening/representative statistics."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


BASE_URL = "https://raw.githubusercontent.com/FujitsuResearch/domain-adaptive-image-detection/main"
FILES = {
    ("general", "global"): "whitening_matrix_general.pt",
    ("general", "local"): "rep_matrix_general.pt",
    ("cars", "global"): "whitening_matrix_cars.pt",
    ("cars", "local"): "rep_matrix_cars.pt",
}


def download_file(filename: str, out_dir: Path, overwrite: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and not overwrite:
        print(f"exists: {out_path}")
        return out_path

    url = f"{BASE_URL}/{filename}"
    print(f"download: {url}")
    urlretrieve(url, out_path)
    print(f"saved: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="networks/weights/clide", help="Output directory.")
    parser.add_argument("--domain", choices=["general", "cars", "all"], default="general")
    parser.add_argument("--kind", choices=["global", "local", "all"], default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    domains = ["general", "cars"] if args.domain == "all" else [args.domain]
    kinds = ["global", "local"] if args.kind == "all" else [args.kind]

    out_dir = Path(args.out_dir)
    for domain in domains:
        for kind in kinds:
            download_file(FILES[(domain, kind)], out_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
