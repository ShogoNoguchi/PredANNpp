#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo.py

PredANN++ Gradio demo for 3-second EEG Song ID classification.

What this demo provides
-----------------------
- Load a finetuned PredANN++ checkpoint (Entropy / Surprisal, or custom path)
- Run inference on ONE sample from NMED-T using the SAME dataloader as training/evaluation:
    codes_3s/predann/datasets/preprocessing_eegmusic_dataset_3s.py
- Show:
    - Ground-truth label (0-9) and corresponding song_id (e.g., 21-30) if available
    - Top-1 predicted label, Top-K list, and ✅/❌ correctness flag
    - Softmax probability bar-plot
    - Quick accuracy on the first N samples (optional)

IMPORTANT (Public Repository)
-----------------------------
- This repository does NOT ship any NMED-T EEG/audio files.
  Please download NMED-T from the official source and point --dataset_dir (or the UI textbox)
  to your local dataset directory.
- We prioritize reproducibility over convenience:
  the demo uses the official training dataloader + deterministic cropping for SW_valid.

License
-------
- This demo script is released under CC-BY-SA 4.0 (see repository LICENSE).
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402


# =============================================================================
# Paths & Imports (use the SAME code as training)
# =============================================================================
REPO_DIR = Path(__file__).resolve().parent
CODES3S_DIR = REPO_DIR / "codes_3s"
if str(CODES3S_DIR) not in sys.path:
    sys.path.insert(0, str(CODES3S_DIR))

# Now we can import project modules
from predann.datasets import get_dataset  # type: ignore  # noqa: E402
from predann.modules.EM_finetune import TransformerEEGEncoder as FinetuneEnc  # type: ignore  # noqa: E402


# =============================================================================
# Logging
# =============================================================================
def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("predannpp_demo")
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    logger.addHandler(sh)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


LOGGER = setup_logger(REPO_DIR / "logs" / "demo.log")


# =============================================================================
# Small utilities
# =============================================================================
def _as_path(p: str) -> Path:
    return Path(str(p)).expanduser().resolve()


def _parse_class_song_id(s: str) -> List[int]:
    """
    The repository stores class_song_id as a string, e.g. "[21,22,23,...,30]".
    This helper converts it to List[int].
    """
    s = str(s).strip()
    if not s.startswith("["):
        raise ValueError(f"class_song_id must look like '[..]'. got={s}")
    s2 = s.strip("[]").strip()
    if not s2:
        return []
    return [int(x.strip()) for x in s2.split(",") if x.strip()]


def _safe_device(device_str: str) -> torch.device:
    device_str = str(device_str).lower().strip()
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# =============================================================================
# Dataset & Model caching
# =============================================================================
@dataclass(frozen=True)
class DatasetKey:
    dataset_dir: str
    window_size: int
    stride: int
    eeg_normalization: str
    clamp_value: int
    eeg_length: int
    audio_clip_length: int
    split_seed: int
    class_song_id: str
    shifting_time: int
    start_position: int


_DATASET_CACHE: Dict[DatasetKey, Any] = {}
_MODEL_CACHE: Dict[Tuple[str, str], Any] = {}  # (ckpt_path, device_str) -> model


def build_sw_valid_dataset(
    dataset_dir: str,
    window_size: int,
    stride: int,
    eeg_normalization: str,
    clamp_value: int,
    eeg_length: int,
    audio_clip_length: int,
    split_seed: int,
    class_song_id: str,
    shifting_time: int,
    start_position: int,
) -> Any:
    """
    Build SW_valid dataset exactly like evaluate.py/main_3s.py do,
    keeping reproducible random numbers AND deterministic cropping.

    NOTE:
    - In preprocessing_eegmusic_dataset_3s.py, __getitem__ uses deterministic=True
      for SW_valid, so the random_numbers list is not used for SW_valid inference.
      However, we still set it to match the official evaluation pipeline.
    """
    ds = get_dataset(
        "preprocessing_eegmusic",
        dataset_dir,
        subset="SW_valid",
        download=False,
    )

    ds.set_sliding_window_parameters(window_size, stride)
    ds.set_eeg_normalization(eeg_normalization, clamp_value)
    ds.set_other_parameters(
        eeg_length,
        audio_clip_length,
        split_seed,
        class_song_id,
        shifting_time,
        start_position=start_position,
    )
    ds.set_mode("Finetune")

    # Reproducible random numbers (even if SW_valid uses deterministic center crop)
    random.seed(42)
    valid_random_numbers = [random.randint(0, max(1, window_size - 375 - 1)) for _ in range(1200)]
    ds.set_random_numbers(valid_random_numbers)

    return ds


def get_or_build_dataset(key: DatasetKey) -> Any:
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    ds = build_sw_valid_dataset(
        dataset_dir=key.dataset_dir,
        window_size=key.window_size,
        stride=key.stride,
        eeg_normalization=key.eeg_normalization,
        clamp_value=key.clamp_value,
        eeg_length=key.eeg_length,
        audio_clip_length=key.audio_clip_length,
        split_seed=key.split_seed,
        class_song_id=key.class_song_id,
        shifting_time=key.shifting_time,
        start_position=key.start_position,
    )
    _DATASET_CACHE[key] = ds
    LOGGER.info(f"[Dataset] built SW_valid: len={len(ds)} dir={key.dataset_dir}")
    return ds


def load_finetune_model(
    ckpt_path: str,
    ds: Any,
    device: torch.device,
    finetune_use_cls_token: int = 1,
) -> FinetuneEnc:

    from argparse import Namespace

    # 🔥 クラス数は dataset.labels() を信用しない
    class_list = _parse_class_song_id(ds.class_song_id)
    num_classes = len(class_list)

    dummy_args = Namespace(
        learning_rate=0.0,
        pretrain_ckpt_path=None,
        finetune_use_cls_token=int(finetune_use_cls_token),
    )

    # 🔥 まず普通にロード
    model = FinetuneEnc.load_from_checkpoint(
        ckpt_path,
        preprocess_dataset=ds,
        args=dummy_args,
        strict=False,
    )

    # 🔥 projector を checkpoint に合わせて再構築
    model.projector1[3] = torch.nn.Linear(
        model.projector1[3].in_features,
        num_classes,
        bias=False
    )

    # 🔥 state_dict を再ロード（projector mismatch 防止）
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)

    model.eval().to(device)
    return model


def get_or_load_model(
    ckpt_path: str,
    ds: Any,
    device: torch.device,
    finetune_use_cls_token: int,
) -> FinetuneEnc:
    ckpt_path = str(_as_path(ckpt_path))
    key = (ckpt_path, str(device))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    model = load_finetune_model(
        ckpt_path=ckpt_path,
        ds=ds,
        device=device,
        finetune_use_cls_token=finetune_use_cls_token,
    )
    _MODEL_CACHE[key] = model
    return model


# =============================================================================
# Inference & Visualization
# =============================================================================
def _build_prob_plot(probs: np.ndarray, title: str) -> Image.Image:
    """
    probs: shape [num_classes]
    """
    fig = plt.figure(figsize=(8.0, 3.5))
    ax = fig.add_subplot(111)

    x = np.arange(len(probs))
    ax.bar(x, probs)
    ax.set_xlabel("Class index (Song ID label: 0-9)")
    ax.set_ylabel("Softmax probability")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)

    return fig_to_pil(fig)


def _format_topk(probs: np.ndarray, k: int) -> str:
    k = int(k)
    idx = np.argsort(probs)[::-1][:k]
    lines = []
    for rank, c in enumerate(idx, start=1):
        lines.append(f"{rank:02d}. class={int(c)}  prob={float(probs[c]):.6f}")
    return "\n".join(lines)


def infer_one_sample(
    ds: Any,
    model: FinetuneEnc,
    sample_index: int,
    device: torch.device,
    top_k: int,
) -> Tuple[str, Image.Image]:
    """
    Returns:
        (text_report, prob_plot_image)
    """
    n = int(sample_index)
    if n < 0 or n >= len(ds):
        raise IndexError(f"sample_index out of range: {n} (len={len(ds)})")

    eeg, label = ds[n]  # Finetune mode returns (eeg, label)
    gt = int(label)

    eeg_b = eeg.unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        cls_hid = model.forward(eeg_b)  # [1, 512]
        logits = model.projector1(model.norm(cls_hid))  # [1, num_classes]
        probs = F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

    pred = int(np.argmax(probs))
    result_flag = "✅ Correct" if pred == gt else "❌ Incorrect"

    # Metadata from dataframe (if available)
    meta_lines: List[str] = []
    try:
        row = ds.df_subset.iloc[n]
        subject = str(row["subject"]) if "subject" in row else "N/A"
        song_label = str(row["song"]) if "song" in row else "N/A"
        trial = str(row["trial"]) if "trial" in row else "N/A"
        chunk = str(row["chunk"]) if "chunk" in row else "N/A"
        window = str(row["window"]) if "window" in row else "N/A"
        eeg_path = str(row["eeg_path"]) if "eeg_path" in row else "N/A"
        audio_path = str(row["audio_path"]) if "audio_path" in row else "N/A"

        meta_lines.append("=== Sample metadata (from dataset df_subset) ===")
        meta_lines.append(f"index  : {n}")
        meta_lines.append(f"subject: {subject}")
        meta_lines.append(f"song(label in df): {song_label}")
        meta_lines.append(f"trial  : {trial}")
        meta_lines.append(f"chunk  : {chunk}")
        meta_lines.append(f"window : {window}")
        meta_lines.append(f"eeg_path  : {eeg_path}")
        meta_lines.append(f"audio_path: {audio_path}")
    except Exception as e:
        meta_lines.append("=== Sample metadata ===")
        meta_lines.append(f"(failed to read df_subset metadata: {e})")

    # Map label -> real song_id (21-30) if ds.class_song_id is available
    real_song_id_gt = None
    real_song_id_pred = None
    try:
        cls_list = _parse_class_song_id(ds.class_song_id)
        if 0 <= gt < len(cls_list):
            real_song_id_gt = cls_list[gt]
        if 0 <= pred < len(cls_list):
            real_song_id_pred = cls_list[pred]
    except Exception:
        real_song_id_gt = None
        real_song_id_pred = None

    report_lines: List[str] = []
    report_lines.append("=== PredANN++ Demo Result ===")
    report_lines.append(f"GT label (0-9): {gt}" + (f"  (song_id={real_song_id_gt})" if real_song_id_gt is not None else ""))
    report_lines.append(f"Pred label    : {pred}" + (f"  (song_id={real_song_id_pred})" if real_song_id_pred is not None else ""))
    report_lines.append(f"Result        : {result_flag}")
    report_lines.append("")
    report_lines.append("=== Top-K ===")
    report_lines.append(_format_topk(probs, top_k))
    report_lines.append("")
    report_lines.extend(meta_lines)

    title = f"Softmax probs (pred={pred}, gt={gt})"
    img = _build_prob_plot(probs, title=title)

    return "\n".join(report_lines), img


def compute_accuracy_first_n(
    ds: Any,
    model: FinetuneEnc,
    device: torch.device,
    n_samples: int,
) -> str:
    n_samples = int(n_samples)
    n_samples = min(n_samples, len(ds))
    if n_samples <= 0:
        return "n_samples must be >= 1"

    correct = 0
    for i in range(n_samples):
        eeg, label = ds[i]
        gt = int(label)

        eeg_b = eeg.unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            cls_hid = model.forward(eeg_b)
            logits = model.projector1(model.norm(cls_hid))
            pred = int(logits.argmax(dim=1).item())

        if pred == gt:
            correct += 1

        if (i + 1) % 50 == 0:
            LOGGER.info(f"[Accuracy] processed {i+1}/{n_samples} samples...")

    acc = float(correct) / float(n_samples)
    return (
        "=== Quick accuracy (SW_valid, deterministic cropping) ===\n"
        f"samples  : {n_samples}\n"
        f"correct  : {correct}\n"
        f"accuracy : {acc:.6f}\n"
        "\nNOTE:\n"
        "- This is a quick sanity-check, not the official evaluation pipeline.\n"
        "- Official evaluation is in: codes_3s/analysis/evaluate.py\n"
    )


# =============================================================================
# UI glue
# =============================================================================
def resolve_ckpt(choice: str, custom_path: str) -> str:
    choice = str(choice).strip()
    if choice == "Entropy (ctx16, seed42)":
        return str(REPO_DIR / "checkpoints" / "PredANNpp_SongAcc_Entropy_ctx16_Pretrain10kEpoch_Finetune3.5kEpoch_seed42.ckpt")
    if choice == "Surprisal (ctx16, seed42)":
        return str(REPO_DIR / "checkpoints" / "PredANNpp_SongAcc_Surprisal_ctx16_Pretrain10kEpoch_Finetune3.5kEpoch_seed42.ckpt")
    if choice == "Custom path":
        return str(custom_path).strip()
    raise ValueError(f"Unknown checkpoint choice: {choice}")


def ui_load_dataset(
    dataset_dir: str,
    window_size: int,
    stride: int,
    eeg_normalization: str,
    clamp_value: int,
    eeg_length: int,
    audio_clip_length: int,
    split_seed: int,
    class_song_id: str,
    shifting_time: int,
    start_position: int,
):
    dataset_dir = str(dataset_dir).strip()
    if not dataset_dir:
        return (
            "ERROR: dataset_dir is empty",
            gr.Slider.update(minimum=0, maximum=0, value=0, step=1),
        )

    key = DatasetKey(
        dataset_dir=str(_as_path(dataset_dir)),
        window_size=int(window_size),
        stride=int(stride),
        eeg_normalization=str(eeg_normalization),
        clamp_value=int(clamp_value),
        eeg_length=int(eeg_length),
        audio_clip_length=int(audio_clip_length),
        split_seed=int(split_seed),
        class_song_id=str(class_song_id),
        shifting_time=int(shifting_time),
        start_position=int(start_position),
    )

    try:
        ds = get_or_build_dataset(key)
        info = (
            "=== Dataset loaded (SW_valid) ===\n"
            f"dataset_dir : {key.dataset_dir}\n"
            f"len(ds)     : {len(ds)}\n"
            f"window_size : {key.window_size}\n"
            f"stride      : {key.stride}\n"
            f"eeg_norm    : {key.eeg_normalization}\n"
            f"clamp_value : {key.clamp_value}\n"
            f"eeg_length  : {key.eeg_length}\n"
            f"audio_clip  : {key.audio_clip_length}\n"
            f"split_seed  : {key.split_seed}\n"
            f"class_song_id: {key.class_song_id}\n"
            f"shifting_time: {key.shifting_time}\n"
            f"start_position: {key.start_position}\n"
        )
        slider = gr.Slider.update(minimum=0, maximum=max(0, len(ds) - 1), value=0, step=1)
        return info, slider
    except Exception as e:
        LOGGER.exception("Failed to load dataset")
        return (
            f"ERROR: failed to load dataset:\n{e}",
            gr.Slider.update(minimum=0, maximum=0, value=0, step=1),
        )


def ui_run_inference(
    dataset_dir: str,
    ckpt_choice: str,
    custom_ckpt_path: str,
    device_str: str,
    finetune_use_cls_token: int,
    sample_index: int,
    top_k: int,
    window_size: int,
    stride: int,
    eeg_normalization: str,
    clamp_value: int,
    eeg_length: int,
    audio_clip_length: int,
    split_seed: int,
    class_song_id: str,
    shifting_time: int,
    start_position: int,
):
    dataset_dir = str(dataset_dir).strip()
    ckpt_path = resolve_ckpt(ckpt_choice, custom_ckpt_path)
    device = _safe_device(device_str)

    key = DatasetKey(
        dataset_dir=str(_as_path(dataset_dir)),
        window_size=int(window_size),
        stride=int(stride),
        eeg_normalization=str(eeg_normalization),
        clamp_value=int(clamp_value),
        eeg_length=int(eeg_length),
        audio_clip_length=int(audio_clip_length),
        split_seed=int(split_seed),
        class_song_id=str(class_song_id),
        shifting_time=int(shifting_time),
        start_position=int(start_position),
    )

    if not os.path.exists(ckpt_path):
        return f"ERROR: checkpoint not found: {ckpt_path}", None

    try:
        ds = get_or_build_dataset(key)
        model = get_or_load_model(
            ckpt_path=ckpt_path,
            ds=ds,
            device=device,
            finetune_use_cls_token=int(finetune_use_cls_token),
        )
        txt, img = infer_one_sample(ds, model, int(sample_index), device, int(top_k))
        header = (
            f"=== Runtime info ===\n"
            f"device   : {device}\n"
            f"ckpt     : {ckpt_path}\n"
            f"use_cls  : {int(finetune_use_cls_token)}\n"
            "\n"
        )
        return header + txt, img
    except Exception as e:
        LOGGER.exception("Inference failed")
        return f"ERROR: inference failed:\n{e}", None


def ui_compute_accuracy(
    dataset_dir: str,
    ckpt_choice: str,
    custom_ckpt_path: str,
    device_str: str,
    finetune_use_cls_token: int,
    n_samples: int,
    window_size: int,
    stride: int,
    eeg_normalization: str,
    clamp_value: int,
    eeg_length: int,
    audio_clip_length: int,
    split_seed: int,
    class_song_id: str,
    shifting_time: int,
    start_position: int,
):
    dataset_dir = str(dataset_dir).strip()
    ckpt_path = resolve_ckpt(ckpt_choice, custom_ckpt_path)
    device = _safe_device(device_str)

    key = DatasetKey(
        dataset_dir=str(_as_path(dataset_dir)),
        window_size=int(window_size),
        stride=int(stride),
        eeg_normalization=str(eeg_normalization),
        clamp_value=int(clamp_value),
        eeg_length=int(eeg_length),
        audio_clip_length=int(audio_clip_length),
        split_seed=int(split_seed),
        class_song_id=str(class_song_id),
        shifting_time=int(shifting_time),
        start_position=int(start_position),
    )

    if not os.path.exists(ckpt_path):
        return f"ERROR: checkpoint not found: {ckpt_path}"

    try:
        ds = get_or_build_dataset(key)
        model = get_or_load_model(
            ckpt_path=ckpt_path,
            ds=ds,
            device=device,
            finetune_use_cls_token=int(finetune_use_cls_token),
        )
        return compute_accuracy_first_n(ds, model, device, int(n_samples))
    except Exception as e:
        LOGGER.exception("Accuracy computation failed")
        return f"ERROR: accuracy computation failed:\n{e}"


def build_ui(default_dataset_dir: str = "") -> gr.Blocks:
    with gr.Blocks(title="PredANN++ Demo (Song ID from EEG)") as demo:
        gr.Markdown(
            "# PredANN++ Demo (Gradio)\n"
            "This demo runs **3-second EEG → Song ID (0-9)** inference using the **official dataloader**.\n\n"
            "**IMPORTANT**: NMED-T dataset is not included in this repository.\n"
            "Please download it from the official source and set `dataset_dir` accordingly.\n"
        )

        with gr.Row():
            dataset_dir = gr.Textbox(
                label="dataset_dir (NMED-T base dir)",
                value=default_dataset_dir,
                placeholder="/path/to/NMED-T_dataset",
            )
            load_btn = gr.Button("Load dataset (SW_valid)")

        with gr.Row():
            ds_info = gr.Textbox(label="Dataset status", lines=10)

        gr.Markdown("## Runtime / Loader Parameters (match training defaults)")
        with gr.Row():
            window_size = gr.Number(label="window_size", value=1000, precision=0)
            stride = gr.Number(label="stride", value=200, precision=0)
            eeg_length = gr.Number(label="eeg_length (3s@125Hz)", value=375, precision=0)
            audio_clip_length = gr.Number(label="audio_clip_length (sec)", value=3, precision=0)

        with gr.Row():
            eeg_normalization = gr.Dropdown(
                label="eeg_normalization",
                choices=["MetaAI", "channel_mean", "all_mean", "constant_multiple"],
                value="MetaAI",
            )
            clamp_value = gr.Number(label="clamp_value (MetaAI)", value=20, precision=0)
            split_seed = gr.Number(label="split_seed", value=42, precision=0)
            shifting_time = gr.Number(label="shifting_time", value=200, precision=0)
            start_position = gr.Number(label="start_position", value=0, precision=0)

        class_song_id = gr.Textbox(
            label="class_song_id (string list)",
            value="[21,22,23,24,25,26,27,28,29,30]",
        )

        gr.Markdown("## Model & Inference")
        with gr.Row():
            ckpt_choice = gr.Dropdown(
                label="Checkpoint",
                choices=["Entropy (ctx16, seed42)", "Surprisal (ctx16, seed42)", "Custom path"],
                value="Entropy (ctx16, seed42)",
            )
            custom_ckpt_path = gr.Textbox(
                label="Custom checkpoint path (used if 'Custom path')",
                value="",
                placeholder="/path/to/your_model.ckpt",
            )

        with gr.Row():
            device_str = gr.Dropdown(label="device", choices=["cuda", "cpu"], value="cuda")
            finetune_use_cls_token = gr.Radio(
                label="finetune_use_cls_token",
                choices=[0, 1],
                value=1,
            )
            top_k = gr.Radio(label="Top-K", choices=[3, 5], value=5)

        sample_index = gr.Slider(
            label="sample_index (SW_valid index)",
            minimum=0,
            maximum=0,
            value=0,
            step=1,
        )

        run_btn = gr.Button("Run inference (one sample)")

        with gr.Row():
            out_text = gr.Textbox(label="Result", lines=30)
        with gr.Row():
            out_img = gr.Image(label="Softmax probability plot", type="pil")

        gr.Markdown("## Quick Accuracy (sanity-check)")
        with gr.Row():
            n_samples = gr.Number(label="n_samples (first N in SW_valid)", value=200, precision=0)
            acc_btn = gr.Button("Compute accuracy")
        acc_text = gr.Textbox(label="Accuracy report", lines=12)

        # Wire actions
        load_btn.click(
            fn=ui_load_dataset,
            inputs=[
                dataset_dir,
                window_size,
                stride,
                eeg_normalization,
                clamp_value,
                eeg_length,
                audio_clip_length,
                split_seed,
                class_song_id,
                shifting_time,
                start_position,
            ],
            outputs=[ds_info, sample_index],
        )

        run_btn.click(
            fn=ui_run_inference,
            inputs=[
                dataset_dir,
                ckpt_choice,
                custom_ckpt_path,
                device_str,
                finetune_use_cls_token,
                sample_index,
                top_k,
                window_size,
                stride,
                eeg_normalization,
                clamp_value,
                eeg_length,
                audio_clip_length,
                split_seed,
                class_song_id,
                shifting_time,
                start_position,
            ],
            outputs=[out_text, out_img],
        )

        acc_btn.click(
            fn=ui_compute_accuracy,
            inputs=[
                dataset_dir,
                ckpt_choice,
                custom_ckpt_path,
                device_str,
                finetune_use_cls_token,
                n_samples,
                window_size,
                stride,
                eeg_normalization,
                clamp_value,
                eeg_length,
                audio_clip_length,
                split_seed,
                class_song_id,
                shifting_time,
                start_position,
            ],
            outputs=[acc_text],
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="PredANN++ Demo (Gradio)")
    parser.add_argument("--dataset_dir", type=str, default="", help="NMED-T base directory")
    parser.add_argument("--server_port", type=int, default=7860)
    parser.add_argument("--server_name", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    ui = build_ui(default_dataset_dir=args.dataset_dir)
    ui.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()