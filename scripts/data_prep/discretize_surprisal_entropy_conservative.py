#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discretize_surprisal_entropy_conservative.py

Conservative pipeline:
Discretize continuous Surprisal and/or Entropy (30s-chunk features) into uint8 codes (Q=128).

Inputs (under <out_root>):
- surprisal_k1/*.npy   (float32, shape=(1500,))
- entropy_k1/*.npy     (float32, shape=(1500,))

Outputs:
- NoClip_Discreat_K1Surprisal/*.npy  (uint8, shape=(1500,))
- Entropy_k1_Q128/*.npy              (uint8, shape=(1500,))

Also saves bin edges in each output folder:
- edges.pkl + edges.json

Method:
- Equal-frequency binning using quantiles across ALL files pooled together.

Usage:
python discretize_surprisal_entropy_conservative.py \
  --out_root /path/to/NMED-T_dataset \
  --feature both
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def collect_npy_files(root: Path) -> List[Path]:
    files = sorted(p for p in root.rglob("*.npy") if p.is_file())
    LOGGER.info("Found %d .npy files under %s", len(files), root)
    if not files:
        raise FileNotFoundError(f"No .npy files found in {root}")
    return files


def load_all_1d(files: List[Path], expected_len: int) -> np.ndarray:
    all_arr: List[np.ndarray] = []
    for f in tqdm(files, desc="Loading", unit="file"):
        arr = np.load(f)
        if arr.dtype != np.float32:
            raise TypeError(f"{f} dtype is {arr.dtype}, expected float32")
        if arr.shape != (expected_len,):
            raise ValueError(f"{f} shape {arr.shape}, expected ({expected_len},)")
        all_arr.append(arr)
    concatenated = np.concatenate(all_arr, axis=0)
    LOGGER.info("Concatenated shape = %s", concatenated.shape)
    return concatenated


def compute_edges(data: np.ndarray, bins: int = 128) -> np.ndarray:
    q = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    edges = np.quantile(data, q)
    eps = np.finfo(edges.dtype).eps
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    LOGGER.info("Computed %d-bin quantile edges", bins)
    return edges.astype(np.float32)


def save_edges(edges: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(edges, f)
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(edges.tolist(), f, ensure_ascii=False, indent=2)
    LOGGER.info("Saved edges to %s (.pkl & .json)", path)


def discretize_and_save(
    files: List[Path],
    edges: np.ndarray,
    input_root: Path,
    output_root: Path,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    bin_edges = edges[1:-1]

    for f in tqdm(files, desc="Discretizing", unit="file"):
        arr = np.load(f).astype(np.float32)
        np.clip(arr, edges[0], edges[-1], out=arr)
        labels = np.digitize(arr, bin_edges, right=False).astype(np.uint8)

        relative = f.relative_to(input_root)
        out_f = output_root / relative
        out_f.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_f, labels)

    LOGGER.info("Finished discretizing %d files -> %s", len(files), output_root)


def run_one_feature(
    out_root: Path,
    feature: str,
    bins: int = 128,
) -> None:
    if feature == "surprisal":
        input_dir = out_root / "surprisal_k1"
        output_dir = out_root / "NoClip_Discreat_K1Surprisal"
        expected_len = 1500
    elif feature == "entropy":
        input_dir = out_root / "entropy_k1"
        output_dir = out_root / "Entropy_k1_Q128"
        expected_len = 1500
    else:
        raise ValueError(f"Unknown feature: {feature}")

    files = collect_npy_files(input_dir)
    pooled = load_all_1d(files, expected_len=expected_len)
    edges = compute_edges(pooled, bins=bins)

    edges_out = output_dir / "edges.pkl"
    save_edges(edges, edges_out)

    discretize_and_save(files, edges, input_root=input_dir, output_root=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservative: discretize Surprisal/Entropy (30s features) into Q=128 quantile bins."
    )
    parser.add_argument("--out_root", type=Path, required=True, help="NMED-T dataset root directory.")
    parser.add_argument("--feature", choices=["surprisal", "entropy", "both"], default="both")
    parser.add_argument("--bins", type=int, default=128)
    args = parser.parse_args()

    out_root = args.out_root.expanduser().resolve()
    if not out_root.exists():
        raise FileNotFoundError(f"out_root not found: {out_root}")

    if args.feature in ("surprisal", "both"):
        run_one_feature(out_root, "surprisal", bins=int(args.bins))
    if args.feature in ("entropy", "both"):
        run_one_feature(out_root, "entropy", bins=int(args.bins))

    LOGGER.info("All done!")


if __name__ == "__main__":
    main()