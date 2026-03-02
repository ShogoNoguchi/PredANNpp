#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_surprisal_entropy.py

Compute MusicGen k1 Surprisal/Entropy with 0.1s stride and 3s segments (full-length songs).

This script processes full-length audio files (~240s) and computes Surprisal/Entropy
using a sliding-window approach:
  - 3-second segments with 0.1-second stride
  - context window (8 / 16 / 32 seconds)
  - Output shape: (N_segments, 150) per song  (150 = 3s / 20ms)

Output structure (per song):
    <out_root>/<song_id>/
        surp.npy   # shape = (N_segments, 150), dtype=float32
        ent.npy    # shape = (N_segments, 150), dtype=float32
        meta.csv   # "segment_idx,segment_start_s,segment_end_s"

Logits cache (optional):
    <logits_dir>/<song_id>/segxxxxx.npz

Requires:
    - audiocraft (https://github.com/facebookresearch/audiocraft)
    - torch, soundfile, numpy, tqdm

Usage Examples
--------------
# Compute both Surprisal and Entropy (8s context)
python compute_surprisal_entropy.py \\
    --audio_dir /path/to/NMED-T_dataset/audio \\
    --out_root /path/to/NMED-T_dataset/SurpEnt0.1stride \\
    --mode both \\
    --window_sec 8.0

# Use 16s context window
python compute_surprisal_entropy.py \\
    --audio_dir /path/to/NMED-T_dataset/audio \\
    --out_root /path/to/NMED-T_dataset/SurpEnt0.1stride_ctx16 \\
    --mode both \\
    --window_sec 16.0

# Recompute from scratch
python compute_surprisal_entropy.py \\
    --audio_dir /path/to/NMED-T_dataset/audio \\
    --out_root /path/to/NMED-T_dataset/SurpEnt0.1stride \\
    --mode both \\
    --overwrite \\
    --refresh_logits
"""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import os
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

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


def setup_logger(name: str = "stride0p1_surp_ent") -> logging.Logger:
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


def get_memory_usage_fraction() -> float:
    """
    Return system memory usage as a fraction (0.0–1.0).

    Used to stop saving logits cache if memory usage is too high.
    """
    try:
        meminfo = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key = parts[0].rstrip(":")
                value_kb = int(parts[1])
                meminfo[key] = value_kb
        total = meminfo.get("MemTotal", None)
        available = meminfo.get("MemAvailable", None)
        if total is None or available is None:
            return 0.0
        used = total - available
        return float(used) / float(total)
    except Exception:
        return 0.0


def load_models(device: torch.device) -> MusicGen:
    """
    Load MusicGen-large and EnCodec (32 kHz, 4-codebook RVQ).
    """
    mg = MusicGen.get_pretrained("facebook/musicgen-large", device=device)
    mg.compression_model.set_num_codebooks(4)

    mg.lm = mg.lm.to(device=device, dtype=torch.float32)
    mg.compression_model = mg.compression_model.to(device=device, dtype=torch.float32)

    mg.lm.eval()
    mg.compression_model.eval()
    return mg


def _read_mono_wav(wav_path: Path) -> Tuple[torch.Tensor, int]:
    """
    Load mono WAV as [1,1,T] tensor with sample rate.
    """
    wav_np, sr = sf.read(str(wav_path), always_2d=False)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    wav = torch.from_numpy(wav_np).to(torch.float32)[None, None, :]
    return wav, sr


def _ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def encode_all_codes(
    wav_path: Path,
    model: MusicGen,
    device: torch.device,
) -> torch.Tensor:
    """
    Encode full WAV to codes via EnCodec: [1, 4, T_full].
    """
    wav, sr = _read_mono_wav(wav_path)
    wav = convert_audio(wav, sr, model.sample_rate, to_channels=1)
    wav = wav.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        codes, _ = model.compression_model.encode(wav)
    return codes.long()


def build_window_codes(
    full_codes: torch.Tensor,
    win_start_frame: int,
    win_len_frames: int,
    special_token_id: int,
) -> torch.Tensor:
    """
    Extract window from full codes, padding with special tokens as needed.
    """
    assert full_codes.dim() == 3
    B, K, T_full = full_codes.shape
    device = full_codes.device

    codes_win = torch.full(
        (B, K, win_len_frames),
        fill_value=int(special_token_id),
        dtype=torch.long,
        device=device,
    )

    src_start = max(win_start_frame, 0)
    src_end = min(win_start_frame + win_len_frames, T_full)

    if src_end > src_start:
        dst_start = src_start - win_start_frame
        dst_end = dst_start + (src_end - src_start)
        codes_win[:, :, dst_start:dst_end] = full_codes[:, :, src_start:src_end]

    return codes_win


def forward_k1_logits_window(
    codes_win: torch.Tensor,
    model: MusicGen,
) -> torch.Tensor:
    """
    Forward codes through LM to get k1 logits: [T_win, card].
    """
    dummy_cond = ConditioningAttributes(text={"description": ""})
    with torch.no_grad():
        lm_out = model.lm.compute_predictions(
            codes_win,
            conditions=[dummy_cond],
            condition_tensors=None,
            keep_only_valid_steps=True,
        )
        logits_k1 = lm_out.logits[0, 0]  # [T_win, card]
    return logits_k1


def entropy_from_logits(logits: torch.Tensor) -> np.ndarray:
    """
    Compute entropy: H_i = -∑_v p_i(v) log p_i(v) (natural log).
    """
    logits = logits.to(torch.float32)
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    ent = -(probs * log_probs).sum(dim=-1)
    return ent.detach().cpu().numpy().astype(np.float32)


def surprisal_from_logits_and_tokens(
    logits: torch.Tensor,
    tokens: torch.Tensor,
) -> np.ndarray:
    """
    Compute surprisal: S_i = -log p_i(true_token) (natural log).
    """
    logits = logits.to(torch.float32)
    tokens = tokens.to(torch.long).to(logits.device)
    T = int(tokens.numel())
    log_probs = torch.log_softmax(logits, dim=-1)
    idx = torch.arange(T, device=logits.device)
    vals = -log_probs[idx, tokens]
    return vals.detach().cpu().numpy().astype(np.float32)


def process_one_song(
    wav_path: Path,
    model: MusicGen,
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> None:
    """
    Process one song: compute Surprisal/Entropy with 3s segments, 0.1s stride.
    """
    stem = wav_path.stem
    logger.info(f"=== Processing song: {stem} ===")

    song_out_dir = Path(args.out_root) / stem
    logits_song_dir = Path(args.logits_dir) / stem
    _ensure_dirs(song_out_dir, logits_song_dir)

    surp_fp = song_out_dir / "surp.npy"
    ent_fp = song_out_dir / "ent.npy"
    meta_fp = song_out_dir / "meta.csv"

    need_surp = args.mode in ("surprisal", "both") and (args.overwrite or not surp_fp.exists())
    need_ent = args.mode in ("entropy", "both") and (args.overwrite or not ent_fp.exists())

    if not need_surp and not need_ent and not args.refresh_logits:
        logger.info(f"[skip] {stem}: already computed")
        return

    # Encode full song
    codes_full = encode_all_codes(wav_path, model, device)
    _, _, T_full = codes_full.shape
    frame_rate = model.compression_model.frame_rate
    logger.info(f"{stem}: frame_rate={frame_rate:.3f} fps, T_full={T_full}")

    # Determine song length
    expected_T = int(round(args.song_duration * frame_rate))
    if T_full < expected_T:
        logger.warning(f"{stem}: T_full={T_full} < expected={expected_T}, using T_full")
        T_song = T_full
    elif T_full > expected_T:
        logger.warning(f"{stem}: T_full={T_full} > expected={expected_T}, truncating")
        codes_full = codes_full[:, :, :expected_T]
        T_song = expected_T
    else:
        T_song = expected_T

    seg_len = int(round(args.segment_sec * frame_rate))       # 3s -> 150
    stride_frames = int(round(args.stride_sec * frame_rate))  # 0.1s -> 5
    win_len = int(round(args.window_sec * frame_rate))        # 8s -> 400 (or 16/32)

    if T_song < seg_len:
        logger.error(f"{stem}: T_song={T_song} < seg_len={seg_len}, skipping")
        return

    n_segments = (T_song - seg_len) // stride_frames + 1
    logger.info(
        f"{stem}: seg={seg_len}, stride={stride_frames}, win={win_len}, "
        f"T_song={T_song}, n_segments={n_segments}"
    )

    surprisal_song = np.zeros((n_segments, seg_len), dtype=np.float32) if need_surp else None
    entropy_song = np.zeros((n_segments, seg_len), dtype=np.float32) if need_ent else None
    meta_rows: List[Tuple[int, float, float]] = []

    special_id = int(model.lm.special_token_id)
    allow_save_logits = True

    for j in range(n_segments):
        seg_start = j * stride_frames
        seg_end = seg_start + seg_len

        win_end = seg_end
        win_start = win_end - win_len

        cache_fp = logits_song_dir / f"seg{j:05d}.npz"

        seg_start_s = seg_start / frame_rate
        seg_end_s = seg_end / frame_rate
        meta_rows.append((j, seg_start_s, seg_end_s))

        tail_logits_np: Optional[np.ndarray] = None

        if cache_fp.exists() and not args.refresh_logits:
            try:
                data = np.load(cache_fp)
                tail_logits_np = data["tail_logits"]
                logger.debug(f"[cache] {stem} seg{j:05d}")
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_fp}: {e}")

        if tail_logits_np is None:
            codes_win = build_window_codes(
                codes_full,
                win_start_frame=win_start,
                win_len_frames=win_len,
                special_token_id=special_id,
            )

            logits_k1 = forward_k1_logits_window(codes_win, model)
            if logits_k1.shape[0] != win_len:
                raise RuntimeError(f"logits shape mismatch: {logits_k1.shape[0]} != {win_len}")

            logits_tail = logits_k1[-seg_len:]
            tail_logits_np = logits_tail.detach().cpu().numpy().astype(np.float32)

            if allow_save_logits and not args.no_save_logits:
                mem_usage = get_memory_usage_fraction()
                if mem_usage >= args.mem_usage_stop_saving:
                    logger.warning(
                        f"Memory usage {mem_usage:.2%} >= {args.mem_usage_stop_saving:.2%}, "
                        "stopping logits cache saving"
                    )
                    allow_save_logits = False
                else:
                    save_dtype = np.float16 if args.logits_dtype == "float16" else np.float32
                    np.savez_compressed(
                        cache_fp,
                        tail_logits=tail_logits_np.astype(save_dtype),
                        win_start=win_start,
                        win_end=win_end,
                        seg_start=seg_start,
                        seg_end=seg_end,
                        seg_idx=j,
                    )
                    logger.info(
                        f"[cache save] {stem} seg{j:05d}: win=[{win_start},{win_end}), "
                        f"seg=[{seg_start},{seg_end}), tail_logits shape={tail_logits_np.shape}, "
                        f"dtype={save_dtype}"
                    )

        tokens_seg = codes_full[0, 0, seg_start:seg_end]
        assert int(tokens_seg.numel()) == seg_len

        logits_tail_torch = torch.from_numpy(tail_logits_np.astype(np.float32))

        if need_ent and entropy_song is not None:
            entropy_song[j, :] = entropy_from_logits(logits_tail_torch)

        if need_surp and surprisal_song is not None:
            surprisal_song[j, :] = surprisal_from_logits_and_tokens(logits_tail_torch, tokens_seg)

        if j in (0, n_segments - 1):
            logger.info(
                f"{stem} seg{j:05d}: frames=[{seg_start},{seg_end}), "
                f"time=[{seg_start_s:.3f}s,{seg_end_s:.3f}s]"
            )

    if need_ent and entropy_song is not None:
        np.save(ent_fp, entropy_song.astype(np.float32))
        logger.info(f"[saved] Entropy: {ent_fp} shape={entropy_song.shape}")

    if need_surp and surprisal_song is not None:
        np.save(surp_fp, surprisal_song.astype(np.float32))
        logger.info(f"[saved] Surprisal: {surp_fp} shape={surprisal_song.shape}")

    with open(meta_fp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["segment_idx", "segment_start_s", "segment_end_s"])
        for seg_idx, s_start, s_end in meta_rows:
            writer.writerow([seg_idx, f"{s_start:.4f}", f"{s_end:.4f}"])
    logger.info(f"[saved] meta: {meta_fp}")


def main(args: argparse.Namespace) -> None:
    logger = setup_logger()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"device = {device}")

    logger.info("Loading MusicGen-large ...")
    model = load_models(device)
    frame_rate = model.compression_model.frame_rate
    logger.info(f"EnCodec frame_rate = {frame_rate} frames/s")

    args.audio_dir = str(Path(args.audio_dir).expanduser().resolve())
    args.out_root = str(Path(args.out_root).expanduser().resolve())
    args.logits_dir = str(Path(args.logits_dir).expanduser().resolve())
    _ensure_dirs(Path(args.out_root), Path(args.logits_dir))

    wav_files = sorted(glob.glob(os.path.join(args.audio_dir, "*.wav")))
    logger.info(f"{len(wav_files)} wave files found in {args.audio_dir}")

    for wav_fp in tqdm(wav_files, desc="songs", ncols=80):
        wav_path = Path(wav_fp)
        try:
            process_one_song(wav_path, model, device, args, logger)
        except Exception as e:
            logger.error(f"Failed on {wav_path}: {e}")
            traceback.print_exc()

    logger.info("=== All done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute MusicGen k1 Surprisal/Entropy with 3s segments and 0.1s stride.\n"
            "This is the default 'newMF' pipeline for PredANN++."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory with full-length WAV files (NMED-T audio/).")
    parser.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root directory. Creates <out_root>/<song_id>/ subdirs.",
    )
    parser.add_argument(
        "--logits_dir",
        type=str,
        default=None,
        help="Directory for logits cache (optional). If not specified, uses <out_root>/../logits_k1_stride0p1",
    )
    parser.add_argument("--mode", choices=["entropy", "surprisal", "both"], default="both")
    parser.add_argument("--song_duration", type=float, default=240.0, help="Expected song duration (seconds).")
    parser.add_argument("--segment_sec", type=float, default=3.0, help="Segment length (seconds).")
    parser.add_argument("--stride_sec", type=float, default=0.1, help="Stride between segments (seconds).")
    parser.add_argument("--window_sec", type=float, default=8.0, help="LM context window (seconds): 8/16/32 are typical.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution (slow).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing surp.npy/ent.npy files.")
    parser.add_argument("--refresh_logits", action="store_true", help="Ignore cached logits and recompute.")
    parser.add_argument("--no_save_logits", action="store_true", help="Disable logits caching.")
    parser.add_argument("--logits_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument(
        "--mem_usage_stop_saving",
        type=float,
        default=0.9,
        help="Stop saving logits when memory usage exceeds this fraction (0.0–1.0).",
    )
    args = parser.parse_args()

    if args.logits_dir is None:
        args.logits_dir = str(Path(args.out_root).parent / "logits_k1_stride0p1")

    main(args)