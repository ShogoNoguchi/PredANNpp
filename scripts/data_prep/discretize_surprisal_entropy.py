#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
discretize_surprisal_entropy.py

Discretize continuous Surprisal/Entropy from newMF (0.1s stride) into discrete codes.

Input:
- <input_root>/<song_id>/surp.npy  (float32, shape=(N_seg, 150))
- <input_root>/<song_id>/ent.npy   (float32, shape=(N_seg, 150))

Output (saved next to each song folder):
- surp_Q128.npy (uint8, shape=(N_seg, 150))
- ent_Q128.npy  (uint8, shape=(N_seg, 150))

Also saves bin edges at the root:
- <input_root>/surp_Q128_edges.pkl + .json
- <input_root>/ent_Q128_edges.pkl  + .json

Method:
- Equal-frequency binning using quantiles across ALL songs pooled together.

Usage:
python discretize_surprisal_entropy.py \
  --input_root /path/to/NMED-T_dataset/SurpEnt0.1stride_ctx16 \
  --feature both
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


def collect_feature_files(root: Path, feature: str) -> List[Path]:
    """
    Collect all <feature>.npy files from <root>/<song_id>/ subdirectories.
    feature: "surp" or "ent"
    """
    files: List[Path] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        f = d / f"{feature}.npy"
        if f.is_file():
            files.append(f)

    LOGGER.info("Found %d '%s.npy' files under %s", len(files), feature, root)
    if not files:
        raise FileNotFoundError(f"No '{feature}.npy' files found under {root}")
    return files


def load_all_values(files: List[Path], expected_T: int = 150) -> np.ndarray:
    """
    Load all files and concatenate into 1D array (pool all files for consistent edges).
    """
    all_arr: List[np.ndarray] = []
    for f in tqdm(files, desc="Loading", unit="file"):
        arr = np.load(f)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] != expected_T:
            raise ValueError(f"{f} shape {arr.shape}, expected (N, {expected_T})")
        all_arr.append(arr.reshape(-1))

    concatenated = np.concatenate(all_arr, axis=0)
    LOGGER.info("Concatenated shape = %s", concatenated.shape)
    return concatenated


def compute_edges(data: np.ndarray, bins: int = 128) -> np.ndarray:
    """
    Compute equal-frequency bin boundaries using quantiles.
    """
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


def discretize_file(
    f: Path,
    edges: np.ndarray,
    feature: str,
    bins: int = 128,
    expected_T: int = 150,
) -> None:
    """
    Discretize one <feature>.npy file to <feature>_Q128.npy.
    """
    arr = np.load(f).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != expected_T:
        raise ValueError(f"{f} shape {arr.shape}, expected (N, {expected_T})")

    # Clip only to global min/max edges to avoid numeric overflow (not an "outlier clipping")
    np.clip(arr, edges[0], edges[-1], out=arr)

    bin_edges = edges[1:-1]
    labels = np.digitize(arr, bin_edges, right=False).astype(np.uint8)

    out_f = f.with_name(f"{feature}_Q128.npy")
    np.save(out_f, labels)
    LOGGER.debug("Saved %s", out_f)


def discretize_feature(root: Path, feature: str, bins: int = 128) -> None:
    files = collect_feature_files(root, feature)
    pooled = load_all_values(files, expected_T=150)
    edges = compute_edges(pooled, bins=bins)

    edges_out = root / f"{feature}_Q128_edges.pkl"
    save_edges(edges, edges_out)

    for f in tqdm(files, desc=f"Discretizing {feature}", unit="file"):
        discretize_file(f, edges, feature=feature, bins=bins, expected_T=150)

    LOGGER.info("Finished discretizing feature='%s' under %s", feature, root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discretize newMF Surprisal/Entropy (0.1s stride) into 128 quantile bins."
    )
    parser.add_argument(
        "--input_root",
        type=Path,
        required=True,
        help="Root directory with <song_id>/<feature>.npy structure (e.g., SurpEnt0.1stride_ctx16).",
    )
    parser.add_argument(
        "--feature",
        type=str,
        choices=["surp", "ent", "both"],
        default="both",
        help="Which feature(s) to discretize.",
    )
    parser.add_argument("--bins", type=int, default=128)
    args = parser.parse_args()

    root = args.input_root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"input_root not found: {root}")

    if args.feature in ("surp", "both"):
        discretize_feature(root, "surp", bins=int(args.bins))
    if args.feature in ("ent", "both"):
        discretize_feature(root, "ent", bins=int(args.bins))

    LOGGER.info("All done!")


if __name__ == "__main__":
    main()