#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_muq.py

Extract MuQ features from 30s audio chunks and optionally discretize via K-means.

This script processes NMED-T 30-second audio chunks through a pre-trained MuQ model
to extract continuous embeddings, then applies K-means clustering (K=128)
to produce discrete feature codes.

Model repository:
- MuQ: https://github.com/tencent-ailab/MuQ/tree/main
Checkpoint (example):
- OpenMuQ/MuQ-large-msd-iter: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter  (Last verified: 2026-02-05)

IMPORTANT:
- The output directory names MUST match the training dataloader:
  codes_3s/predann/datasets/preprocessing_eegmusic_dataset_3s.py expects:
    - MuQ_Continuous_embedding/
    - MuQ_Discreat_K128/   (NOTE: 'Discreat' spelling is kept for backward compatibility)

Usage Examples
--------------
# Extract embeddings and discretize (full pipeline)
python process_muq.py \\
    --mode all \\
    --audio_dir /path/to/NMED-T_dataset/audio_30s \\
    --out_root /path/to/NMED-T_dataset \\
    --muq_checkpoint_dir /path/to/MuQ/ckpt \\
    --device cuda

# Only extract continuous embeddings
python process_muq.py \\
    --mode extract \\
    --audio_dir /path/to/NMED-T_dataset/audio_30s \\
    --out_root /path/to/NMED-T_dataset \\
    --muq_checkpoint_dir /path/to/MuQ/ckpt

# Only discretize (assumes embeddings already exist)
python process_muq.py \\
    --mode kmeans \\
    --out_root /path/to/NMED-T_dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import librosa
from sklearn.cluster import KMeans
from tqdm import tqdm


# =====================================================================
# Logger Setup
# =====================================================================
def setup_logger(name: str = "process_muq") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)
    logger.propagate = False
    return logger


logger = setup_logger()


# =====================================================================
# MuQ Module Loading
# =====================================================================
def _load_muq_module(muq_src_path: Optional[Path] = None) -> Tuple[object, object]:
    """
    Import MuQ module, handling a custom source path.

    Args:
        muq_src_path:
            Path to the 'src' directory of the MuQ repository:
            https://github.com/tencent-ailab/MuQ/tree/main/src

    Returns:
        (MuQ, MuQConfig) classes from the muq module.
    """
    if muq_src_path:
        muq_src_path = Path(muq_src_path).expanduser().resolve()
        if str(muq_src_path) not in sys.path:
            sys.path.insert(0, str(muq_src_path))
            logger.info(f"Added MuQ source path: {muq_src_path}")

    try:
        from muq import MuQ, MuQConfig  # type: ignore
        return MuQ, MuQConfig
    except ImportError as e:
        raise ImportError(
            "Failed to import MuQ.\n\n"
            "Options:\n"
            "  (A) Install muq if available in your environment\n"
            "  (B) Clone https://github.com/tencent-ailab/MuQ and pass --muq_src_path <MuQ_repo>/src\n"
        ) from e


# =====================================================================
# WeightNorm key fix
# =====================================================================
def _fix_weightnorm_keys(state_dict: dict) -> dict:
    """
    Fix weight normalization parameter names:
    - parametrizations.weight.original0 -> weight_g
    - parametrizations.weight.original1 -> weight_v
    """
    new_sd = {}
    for k, v in state_dict.items():
        if "parametrizations.weight.original0" in k:
            k = k.replace("parametrizations.weight.original0", "weight_g")
        elif "parametrizations.weight.original1" in k:
            k = k.replace("parametrizations.weight.original1", "weight_v")
        new_sd[k] = v
    return new_sd


# =====================================================================
# MuQ Model Loading
# =====================================================================
def load_muq(
    cfg_path: Path,
    ckpt_path: Path,
    device: torch.device,
    MuQ: object,
    MuQConfig: object,
):
    """
    Load pre-trained MuQ model in eval mode.

    Args:
        cfg_path: Path to config.json
        ckpt_path: Path to pytorch_model.bin
        device: torch device
        MuQ: MuQ class
        MuQConfig: MuQConfig class

    Returns:
        muq model in eval mode, gradients disabled
    """
    logger.info(f"Loading MuQ config: {cfg_path}")
    cfg_dict = json.loads(cfg_path.read_text())
    muq_cfg = MuQConfig(**cfg_dict)
    muq = MuQ(muq_cfg).to(device)

    logger.info(f"Loading MuQ checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location="cpu")
    state_dict = _fix_weightnorm_keys(state_dict)

    missing, unexpected = muq.load_state_dict(state_dict, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch:\n  missing={missing}\n  unexpected={unexpected}"
        )

    muq.eval()
    muq.requires_grad_(False)
    logger.info("MuQ model loaded and frozen")
    return muq


# =====================================================================
# Audio Loading
# =====================================================================
def load_audio(
    wav_path: Path,
    target_sr: int = 24_000,
    expected_sec: int = 30,
) -> torch.Tensor:
    """
    Load WAV and resample to 24 kHz, pad/trim to exactly expected_sec seconds.

    Returns:
        Tensor [1, target_sr*expected_sec] float32
    """
    wav_np, sr = librosa.load(str(wav_path), sr=None, mono=True)

    if sr != target_sr:
        wav_np = librosa.resample(
            wav_np,
            orig_sr=sr,
            target_sr=target_sr,
            res_type="kaiser_fast",
        )

    total_len = target_sr * expected_sec
    if wav_np.shape[0] < total_len:
        wav_np = np.pad(wav_np, (0, total_len - wav_np.shape[0]), mode="constant")
    elif wav_np.shape[0] > total_len:
        wav_np = wav_np[:total_len]

    wav_np = wav_np.astype(np.float32, copy=False)
    return torch.from_numpy(wav_np).unsqueeze(0)  # [1, T]


# =====================================================================
# Extraction: 30s Audio → MuQ Embeddings
# =====================================================================
def process_one_file(
    wav_path: Path,
    embed_path: Path,
    muq: object,
    device: torch.device,
) -> None:
    """
    Extract MuQ embeddings from a single 30s WAV file.

    Output:
        embed_path: .npy (shape: [750, 1024])
    """
    embed_path.parent.mkdir(parents=True, exist_ok=True)

    if embed_path.exists():
        logger.info(f"Skip (exists): {embed_path}")
        return

    audio = load_audio(wav_path, target_sr=24_000, expected_sec=30).to(device)

    with torch.no_grad():
        _, hid = muq.model.get_predictions(audio, is_features_only=False)
        muq_hid = hid[-1].cpu().numpy()  # (1, 750, 1024)

    np.save(embed_path, muq_hid.squeeze(0).astype(np.float32))
    logger.info("Saved MuQ embed: %s  shape=%s", embed_path, muq_hid.shape[1:])


def extract_all_embeddings(
    audio_dir: Path,
    embed_dir: Path,
    muq: object,
    device: torch.device,
) -> List[Path]:
    """
    Extract embeddings for all WAV files in audio_dir.

    Returns:
        List of saved embedding paths
    """
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files in {audio_dir}")

    logger.info(f"Found {len(wav_files)} WAV files, extracting embeddings...")
    saved_paths: List[Path] = []

    for wav in tqdm(wav_files, desc="Extract embeddings", ncols=80):
        npy_name = wav.with_suffix(".npy").name
        embed_path = embed_dir / npy_name
        process_one_file(wav, embed_path, muq, device)
        saved_paths.append(embed_path)

    return saved_paths


# =====================================================================
# Discretization: Embeddings → K-means Codes
# =====================================================================
def discretize_embeddings(
    embed_dir: Path,
    discrete_dir: Path,
    n_clusters: int = 128,
    random_state: int = 0,
    centroid_path: Optional[Path] = None,
) -> None:
    """
    Apply K-means to continuous embeddings and save discrete codes.

    - Loads all [750, 1024] embeddings
    - Fits K-means (K=128)
    - Saves centroids (optional)
    - Saves discrete codes [750] uint8 for each file
    """
    embed_paths = sorted(embed_dir.glob("*.npy"))
    if not embed_paths:
        raise FileNotFoundError(f"No embeddings in {embed_dir}")

    logger.info("Stacking all embeddings for K-means...")
    all_hid = np.concatenate([np.load(p, mmap_mode="r") for p in embed_paths], axis=0)
    logger.info(f"Total frames: {all_hid.shape[0]}, dim: {all_hid.shape[1]}")

    logger.info(f"Fitting K-means (K={n_clusters})...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,  # keep compatibility with older scikit-learn
        random_state=random_state,
        verbose=1,
    ).fit(all_hid)

    if centroid_path:
        centroid_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(centroid_path, kmeans.cluster_centers_.astype(np.float32))
        logger.info(f"Centroids saved: {centroid_path} shape={kmeans.cluster_centers_.shape}")

    logger.info("Discretizing each embedding file...")
    discrete_dir.mkdir(parents=True, exist_ok=True)

    for embed_path in tqdm(embed_paths, desc="Discretize", ncols=80):
        hid = np.load(embed_path)  # [750, 1024]
        ids = kmeans.predict(hid)  # [750]

        disc_path = discrete_dir / embed_path.name
        disc_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(disc_path, ids.astype(np.uint8))
        logger.info("Discrete ID saved: %s", disc_path)


# =====================================================================
# Main
# =====================================================================
def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"device = {device}")

    audio_dir = Path(args.audio_dir).expanduser().resolve() if args.audio_dir else None
    out_root = Path(args.out_root).expanduser().resolve()

    embed_dir = out_root / "MuQ_Continuous_embedding"
    # IMPORTANT: keep backward-compatible spelling "Discreat" to match dataloader
    discrete_dir = out_root / "MuQ_Discreat_K128"
    centroid_path = discrete_dir / "muq_k128_centroids.npy"

    logger.info(f"Output root: {out_root}")
    logger.info(f"Embedding dir: {embed_dir}")
    logger.info(f"Discrete  dir: {discrete_dir}")

    # ==================== EXTRACT ====================
    if args.mode in ("extract", "all"):
        if not audio_dir:
            raise ValueError("--audio_dir is required for extract/all mode")

        logger.info("=== EXTRACT Mode ===")
        logger.info(f"Audio dir: {audio_dir}")

        MuQ, MuQConfig = _load_muq_module(args.muq_src_path)

        ckpt_dir = Path(args.muq_checkpoint_dir).expanduser().resolve()
        cfg_path = ckpt_dir / "config.json"
        ckpt_path = ckpt_dir / "pytorch_model.bin"

        if not cfg_path.exists():
            raise FileNotFoundError(f"MuQ config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"MuQ checkpoint not found: {ckpt_path}")

        muq = load_muq(cfg_path, ckpt_path, device, MuQ, MuQConfig)
        extract_all_embeddings(audio_dir, embed_dir, muq, device)
        logger.info("Extraction done")

    # ==================== KMEANS ====================
    if args.mode in ("kmeans", "all"):
        logger.info("=== KMEANS Mode ===")
        discretize_embeddings(
            embed_dir=embed_dir,
            discrete_dir=discrete_dir,
            n_clusters=128,
            random_state=0,
            centroid_path=centroid_path,
        )
        logger.info("K-means done")

    logger.info("=== All done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract MuQ features and discretize via K-means.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["extract", "kmeans", "all"],
        default="all",
        help=(
            "'extract': run feature extraction only\n"
            "'kmeans': run K-means discretization only (requires existing embeddings)\n"
            "'all': run both (default)"
        ),
    )
    parser.add_argument(
        "--audio_dir",
        type=str,
        default=None,
        help="Directory with 30s WAV chunks (required for extract/all). Output from save_nmedt_audio_30s.py",
    )
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument(
        "--muq_checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing MuQ checkpoint files: config.json and pytorch_model.bin",
    )
    parser.add_argument(
        "--muq_src_path",
        type=str,
        default=None,
        help="Path to MuQ 'src' directory if MuQ is not installed as a package.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    main(args)