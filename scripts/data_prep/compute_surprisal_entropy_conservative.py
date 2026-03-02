#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_surprisal_entropy_conservative.py

Conservative pipeline:
Compute MusicGen k1 Surprisal / Entropy for 30-second NMED-T audio chunks.

This script expects 30s chunked WAV files, typically produced by:
    save_nmedt_audio_30s.py

It uses MusicGen-large (Audiocraft) to obtain k1 logits and compute:
- Surprisal (negative log-probability of the true token)
- Entropy   (distribution entropy)

Outputs:
- <out_root>/surprisal_k1/<stem>.npy   (float32, shape=(1500,))
- <out_root>/entropy_k1/<stem>.npy     (float32, shape=(1500,))
Optionally caches logits:
- <out_root>/logits_k1/<stem>.k1logits.npz

Usage Examples
--------------
python compute_surprisal_entropy_conservative.py \\
  --audio_dir <NMEDT_BASE_DIR>/audio_30s \\
  --out_root  <NMEDT_BASE_DIR> \\
  --mode both
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

try:
    from audiocraft.models import MusicGen
    from audiocraft.data.audio_utils import convert_audio
    from audiocraft.modules.conditioners import ConditioningAttributes
except ImportError as e:
    raise ImportError(
        "audiocraft is required.\n"
        "Install via:\n"
        "  pip install audiocraft\n"
        "or clone from https://github.com/facebookresearch/audiocraft\n"
    ) from e


def setup_logger(name: str = "surprisal_entropy_k1_conservative") -> logging.Logger:
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


def load_models(device: torch.device) -> MusicGen:
    mg = MusicGen.get_pretrained("facebook/musicgen-large", device=device)
    mg.compression_model.set_num_codebooks(4)
    mg.lm = mg.lm.to(device=device, dtype=torch.float32)
    mg.compression_model = mg.compression_model.to(device=device, dtype=torch.float32)
    mg.lm.eval()
    mg.compression_model.eval()
    return mg


def _read_mono_wav(wav_path: Path) -> Tuple[torch.Tensor, int]:
    wav_np, sr = sf.read(str(wav_path), always_2d=False)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    wav = torch.from_numpy(wav_np).to(torch.float32)[None, None, :]
    return wav, sr


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def encode_k1_codes(
    wav_path: Path,
    model: MusicGen,
    device: torch.device,
) -> torch.Tensor:
    wav, sr = _read_mono_wav(wav_path)
    wav = convert_audio(wav, sr, model.sample_rate, to_channels=1)
    wav = wav.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        codes, _ = model.compression_model.encode(wav)  # [1, 4, T]
    return codes.long()


def forward_k1_logits(
    codes: torch.Tensor,
    model: MusicGen,
) -> torch.Tensor:
    dummy_cond = ConditioningAttributes(text={"description": ""})
    with torch.no_grad():
        lm_out = model.lm.compute_predictions(
            codes,
            conditions=[dummy_cond],
            condition_tensors=None,
            keep_only_valid_steps=True,
        )
        logits_k1 = lm_out.logits[0, 0]  # [T, card]
    return logits_k1


def compute_logits_and_tokens(
    wav_path: Path,
    model: MusicGen,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    codes = encode_k1_codes(wav_path, model, device)  # [1,4,T]
    logits_k1 = forward_k1_logits(codes, model)       # [T,card]
    tokens_k1 = codes[0, 0]                           # [T]
    return logits_k1, tokens_k1


def entropy_from_logits(logits: torch.Tensor) -> np.ndarray:
    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    probs = torch.exp(log_probs)
    ent = -(probs * log_probs).sum(dim=-1)
    return ent.cpu().numpy().astype(np.float32)


def surprisal_from_logits_and_tokens(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> np.ndarray:
    T = tokens.numel()
    log_probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
    idx = torch.arange(T, device=logits.device)
    val = -log_probs[idx, tokens.to(torch.long)]
    return val.detach().cpu().numpy().astype(np.float32)


def main(args: argparse.Namespace) -> None:
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"device = {device}")

    logger.info("Loading MusicGen-large ...")
    model = load_models(device)
    frame_rate = model.compression_model.frame_rate
    expected_T = int(30.0 * frame_rate)
    logger.info(f"expected_T (30s @ {frame_rate} fps) = {expected_T}")

    audio_dir = Path(args.audio_dir).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    surprisal_out = out_root / "surprisal_k1"
    entropy_out = out_root / "entropy_k1"
    logits_cache = out_root / "logits_k1"
    _ensure_dirs(surprisal_out, entropy_out, logits_cache)

    logger.info(f"Audio dir: {audio_dir}")
    logger.info(f"Output root: {out_root}")
    logger.info(f"  Surprisal -> {surprisal_out}")
    logger.info(f"  Entropy   -> {entropy_out}")
    logger.info(f"  Logits cache -> {logits_cache}")

    wav_files = sorted(glob.glob(str(audio_dir / "*.wav")))
    if not wav_files:
        logger.warning(f"No .wav files found in {audio_dir}")
        return

    logger.info(f"{len(wav_files)} wave files found")

    for wav_fp in tqdm(wav_files, desc="processing", ncols=80):
        wav_path = Path(wav_fp)
        stem = wav_path.stem

        logits_fp = logits_cache / f"{stem}.k1logits.npz"
        surp_fp = surprisal_out / f"{stem}.npy"
        ent_fp = entropy_out / f"{stem}.npy"

        logits_np: Optional[np.ndarray] = None
        tokens_k1: Optional[torch.Tensor] = None
        computed_now = False

        if logits_fp.exists() and not args.refresh_logits:
            try:
                logits_np = np.load(logits_fp)["logits"]
                logger.info(f"[cache hit] {logits_fp} shape={logits_np.shape} dtype={logits_np.dtype}")
            except Exception as e:
                logger.warning(f"Failed to load cached logits ({logits_fp}): {e}, recomputing...")

        if logits_np is None:
            try:
                logits_k1, tokens_k1 = compute_logits_and_tokens(wav_path, model, device)

                save_dtype = np.float16 if args.logits_dtype == "float16" else np.float32
                np.savez_compressed(logits_fp, logits=logits_k1.cpu().numpy().astype(save_dtype))
                logger.info(f"[cache save] {stem}: dtype={save_dtype}")

                logits_np = logits_k1.cpu().numpy()
                computed_now = True
            except Exception as e:
                logger.error(f"Failed LM forward on {wav_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        T = int(logits_np.shape[0])
        if T != expected_T:
            logger.error(f"Unexpected T={T} (expected {expected_T}) for {stem}")
            continue

        if args.mode in ("entropy", "both"):
            if (not ent_fp.exists()) or args.overwrite:
                ent = entropy_from_logits(torch.from_numpy(logits_np))
                np.save(ent_fp, ent.astype(np.float32))
                logger.info(f"Entropy saved: {ent_fp}")
            else:
                logger.info(f"[skip] Entropy exists: {stem}")

        if args.mode in ("surprisal", "both"):
            if (not surp_fp.exists()) or args.overwrite:
                if tokens_k1 is None:
                    try:
                        codes = encode_k1_codes(wav_path, model, device)
                        tokens_k1 = codes[0, 0]
                    except Exception as e:
                        logger.error(f"Failed to encode tokens for {stem}: {e}")
                        continue

                surp = surprisal_from_logits_and_tokens(torch.from_numpy(logits_np), tokens_k1)
                np.save(surp_fp, surp.astype(np.float32))
                logger.info(f"Surprisal saved: {surp_fp}")
            else:
                logger.info(f"[skip] Surprisal exists: {stem}")

        if computed_now:
            logger.info(f"{stem}: card={int(logits_np.shape[1])}, T={T}")

    logger.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Conservative: compute MusicGen k1 Surprisal/Entropy for 30s WAV chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing 30s WAV chunks (audio_30s).")
    parser.add_argument("--out_root", type=str, required=True, help="NMED-T dataset root directory.")
    parser.add_argument("--mode", choices=["entropy", "surprisal", "both"], default="both")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--refresh_logits", action="store_true")
    parser.add_argument("--logits_dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    main(args)