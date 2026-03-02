#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_nmedt_audio_30s.py

Split NMED-T audio files (44.1 kHz) into 30-second chunks.

Input: NMED-T/audio/{21.wav, 22.wav, ..., 30.wav} 
Output: audio_30s/{21_chunk0.wav, 21_chunk1.wav, ..., 30_chunk7.wav}

Usage
-----
python save_nmedt_audio_30s.py \\
    --nmed_root /path/to/NMED-T_dataset \\
    --out_root /path/to/NMED-T_dataset
"""

import argparse
import logging
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("save30s")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
    h = logging.StreamHandler()
    h.setFormatter(fmt)
    logger.addHandler(h)
    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Split NMED-T audio into 30s chunks."
    )
    parser.add_argument(
        "--nmed_root",
        required=True,
        help="Path to NMED-T_dataset (audio/ subdirectory expected).",
    )
    parser.add_argument(
        "--out_root",
        required=True,
        help="Output root (will create/use audio_30s/ subdirectory).",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=44100,
        help="Sample rate (Hz). Default: 44100.",
    )
    parser.add_argument(
        "--chunk_sec",
        type=int,
        default=30,
        help="Chunk duration (seconds). Default: 30.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=8,
        help="Max chunks per song. Default: 8.",
    )
    args = parser.parse_args()

    logger = setup_logger()

    audio_dir = Path(args.nmed_root) / "audio"
    out_dir = Path(args.out_root) / "audio_30s"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(
        audio_dir.glob("*.wav"),
        key=lambda p: int(p.stem.split("_")[0] if "_" in p.stem else p.stem)
    )

    if not wav_files:
        logger.error(f"No .wav files found in {audio_dir}")
        return

    logger.info(f"Processing {len(wav_files)} songs from {audio_dir}")
    sr = args.sample_rate
    chunk_T = sr * args.chunk_sec

    for wav_path in tqdm(wav_files, desc="Split audio", ncols=80):
        song_id = int(wav_path.stem.split("_")[0] if "_" in wav_path.stem else wav_path.stem)

        wav, wav_sr = torchaudio.load(str(wav_path))  # [1, T]
        if wav_sr != sr:
            logger.error(f"{wav_path.name}: sample_rate {wav_sr} ≠ {sr}")
            continue

        total_T = wav.shape[1]
        n_chunks = min(total_T // chunk_T, args.max_chunks)

        if n_chunks < args.max_chunks:
            logger.warning(f"{wav_path.name}: only {n_chunks} of {args.max_chunks} chunks available")

        for c in range(n_chunks):
            s = c * chunk_T
            e = s + chunk_T
            chunk_wav = wav[:, s:e]

            out_path = out_dir / f"{song_id}_chunk{c}.wav"
            torchaudio.save(str(out_path), chunk_wav, sr, encoding="PCM_S", bits_per_sample=16)

    logger.info("✓ Audio chunking complete")


if __name__ == "__main__":
    main()
