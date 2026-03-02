#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation with McNemar statistical testing for PredANN+ (single + ensemble).

Evaluates single models and equal-weight ensembles on SW_valid:
- Single model accuracy with logits caching
- Equal-weight ensembles (2, 3, 4 model combinations)
- McNemar exact test at 4 granularities (all/subject/song/subject×song)
- Complementary metrics: R (disagreement rate), Acc∪, φ (phi coefficient)

Usage:
    python evaluate.py \
        --ckpt_dir /path/to/best_checkpoints \
        --out_dir /path/to/output \
        --mode checkpoint

Models:
- Fullscratch (multiple seeds)
- MuQMultitask → Finetune
- SurpMultitask → Finetune
- EntropyMultitask → Finetune
"""

import os
import sys
import json
import argparse
import logging
import datetime
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from statsmodels.stats.contingency_tables import mcnemar

# Add parent directory to path
CODES3S_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODES3S_DIR))

from predann.utils.yaml_config_hook import yaml_config_hook
from predann.datasets import get_dataset
from predann.modules.EM_finetune import TransformerEEGEncoder as FinetuneEnc


# ============================================================================
# Argument parsing
# ============================================================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluation with McNemar testing (PredANN+)"
    )
    
    # Load default config
    cfg = yaml_config_hook(CODES3S_DIR / "config" / "config.yaml")
    for k, v in cfg.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    
    # I/O paths (user-specified)
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Directory containing model checkpoints")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for results and cache")
    
    # Checkpoint specifications
    parser.add_argument("--fullscratch_seeds", type=str, default="42",
                        help="Comma-separated seed values for Fullscratch models (e.g., '42,1,2,3')")
    parser.add_argument("--multitask_seeds", type=str, default="42",
                        help="Comma-separated seed values for Multitask→Finetune models")
    
    # Execution mode
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--mode", choices=["checkpoint", "offline"],
                   help="checkpoint: infer if cache missing | offline: use cache only")
    
    # Other
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    
    return parser.parse_args()


# ============================================================================
# Logging setup
# ============================================================================
def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "evaluate.log"
    
    logger = logging.getLogger("evaluate")
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Console handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    
    # File handler
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    
    logger.addHandler(sh)
    logger.addHandler(fh)
    
    return logger


# ============================================================================
# Checkpoint path resolution
# ============================================================================
def build_ckpt_map(args) -> Dict[str, str]:
    """Build checkpoint path mapping for all models."""
    ckpt_dir = Path(args.ckpt_dir)
    ckpts = {}
    
    # Fullscratch models (multiple seeds)
    fullscratch_seeds = [s.strip() for s in args.fullscratch_seeds.split(",")]
    for seed in fullscratch_seeds:
        key = f"Fullscratch_seed{seed}"
        # Assume structure: ckpt_dir/Fullscratch_seed{seed}/last.ckpt
        path = ckpt_dir / f"Fullscratch_seed{seed}" / "last.ckpt"
        if path.exists():
            ckpts[key] = str(path)
    
    # Multitask → Finetune models
    multitask_seeds = [s.strip() for s in args.multitask_seeds.split(",")]
    for mode in ["MuQMultitask", "SurpMultitask", "EntropyMultitask"]:
        for seed in multitask_seeds:
            key = f"{mode}_Finetune_seed{seed}"
            # Assume structure: ckpt_dir/{mode}_Finetune_seed{seed}/last.ckpt
            path = ckpt_dir / f"{mode}_Finetune_seed{seed}" / "last.ckpt"
            if path.exists():
                ckpts[key] = str(path)
    
    return ckpts


# ============================================================================
# DataLoader
# ============================================================================
def build_dataloader(args):
    """Build SW_valid dataloader with reproducible random cropping."""
    ds = get_dataset(
        args.dataset,
        args.dataset_dir,
        subset="SW_valid",
        download=False
    )
    
    ds.set_sliding_window_parameters(args.window_size, args.stride)
    ds.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    ds.set_other_parameters(
        args.eeg_length,
        args.audio_clip_length,
        args.split_seed,
        args.class_song_id,
        args.shifting_time,
        start_position=0
    )
    ds.set_mode("Finetune")
    
    # Reproducible random cropping for SW_valid
    random.seed(42)
    valid_random_numbers = [
        random.randint(0, args.window_size - 375 - 1) for _ in range(1200)
    ]
    ds.set_random_numbers(valid_random_numbers)
    
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False
    )
    
    return ds, loader


# ============================================================================
# Cache I/O
# ============================================================================
def cache_path(out_dir: Path, model_key: str) -> Path:
    cache_dir = out_dir / "logits" / model_key
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "songid_cache.pt"


def save_cache(data: dict, path: Path):
    torch.save(data, path)


def load_cache(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path}")
    return torch.load(path, map_location="cpu")


# ============================================================================
# Inference
# ============================================================================
@torch.no_grad()
def inference_loop(model, loader, device="cuda"):
    """Run inference and collect logits, labels, and metadata."""
    model.eval().to(device)
    
    logits_list, lbl_list = [], []
    subj_list, song_list = [], []
    offset = 0
    
    for batch in loader:
        eeg = batch[0].to(device)
        label = batch[1].to(device)
        
        B = eeg.size(0)
        eeg3 = eeg.view(B, 128, 3, 125)
        
        # Forward pass
        cls_hid = model.emenc(eeg3)
        cls_logits = model.projector1(model.norm(cls_hid))
        
        logits_list.append(cls_logits.detach().cpu())
        lbl_list.append(label.detach().cpu())
        
        # Extract metadata from dataset
        subj_arr = loader.dataset.df_subset["subject"].values[offset:offset+B]
        song_arr = loader.dataset.df_subset["song"].values[offset:offset+B]
        subj_list.append(subj_arr)
        song_list.append(song_arr)
        offset += B
    
    return {
        "cls_logits": torch.cat(logits_list, dim=0),
        "cls_lbl": torch.cat(lbl_list, dim=0),
        "subject": np.concatenate(subj_list),
        "song": np.concatenate(song_list),
    }


def load_or_infer(model_key: str, ckpt_path: str, args, out_dir: Path, 
                  ds, loader, logger):
    """Load cached logits or run inference."""
    cache_file = cache_path(out_dir, model_key)
    
    # Use cache if available in checkpoint mode
    if cache_file.exists() and args.mode == "checkpoint":
        logger.info(f"[{model_key}] Loading cache: {cache_file}")
        return load_cache(cache_file)
    
    # Offline mode requires existing cache
    if args.mode == "offline":
        logger.info(f"[{model_key}] Loading cache (offline): {cache_file}")
        return load_cache(cache_file)
    
    # Run inference
    logger.info(f"[{model_key}] Running inference from: {ckpt_path}")
    
    from argparse import Namespace
    dummy_args = Namespace(**{**vars(args), "learning_rate": 0.0, "device": args.device})
    dummy_args.fullscratch_ckpt_path = ckpt_path
    
    model = FinetuneEnc.load_from_checkpoint(
        ckpt_path,
        preprocess_dataset=ds,
        args=dummy_args,
        strict=False
    )
    
    result = inference_loop(model, loader, device=args.device)
    
    # Save cache
    save_cache(result, cache_file)
    logger.info(f"[{model_key}] Cache saved: {cache_file}")
    
    return result


# ============================================================================
# Metrics
# ============================================================================
def accuracy(pred: np.ndarray, lbl: np.ndarray) -> float:
    return float((pred == lbl).mean())


def contingency_table(correct_a: np.ndarray, correct_b: np.ndarray) -> Tuple[int, int, int, int]:
    """Compute 2x2 contingency table for McNemar test."""
    a = int(np.logical_and(correct_a,  correct_b).sum())
    b = int(np.logical_and(correct_a, ~correct_b).sum())
    c = int(np.logical_and(~correct_a,  correct_b).sum())
    d = int(np.logical_and(~correct_a, ~correct_b).sum())
    return a, b, c, d


def phi_coefficient(a: int, b: int, c: int, d: int) -> float:
    """Compute phi coefficient from contingency table."""
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return 0.0
    return (a * d - b * c) / math.sqrt(float(denom))


def mcnemar_with_metrics(pred_a: np.ndarray, pred_b: np.ndarray, lbl: np.ndarray) -> dict:
    """McNemar exact test with complementary metrics."""
    ca = (pred_a == lbl)
    cb = (pred_b == lbl)
    
    a, b, c, d = contingency_table(ca, cb)
    
    # McNemar exact test
    p_value = mcnemar([[a, b], [c, d]], exact=True).pvalue if (b + c) > 0 else 1.0
    
    # Better model (p < 0.05)
    better = "tie" if p_value >= 0.05 else ("A" if b > c else "B")
    
    # Complementary metrics
    N = a + b + c + d
    R = (b + c) / N if N > 0 else 0.0  # Disagreement rate
    acc_union = (a + b + c) / N if N > 0 else 0.0  # Union accuracy
    phi = phi_coefficient(a, b, c, d)
    
    return {
        "a": a, "b": b, "c": c, "d": d,
        "p_value": float(p_value),
        "better": better,
        "R": R,
        "acc_union": acc_union,
        "phi": phi
    }


def group_indices(meta: dict, granularity: str) -> Dict[str, np.ndarray]:
    """Group sample indices by granularity."""
    if granularity == "all":
        return {"__all__": np.arange(len(meta["cls_lbl"]))}
    
    if granularity == "subject":
        vals = meta["subject"]
        return {str(s): np.where(vals == s)[0] for s in np.unique(vals)}
    
    if granularity == "song":
        vals = meta["song"]
        return {str(s): np.where(vals == s)[0] for s in np.unique(vals)}
    
    if granularity == "subject_song":
        subs = meta["subject"]
        songs = meta["song"]
        groups = {}
        for i, (u, v) in enumerate(zip(subs, songs)):
            key = f"({u},{v})"
            if key not in groups:
                groups[key] = []
            groups[key].append(i)
        return {k: np.array(v, dtype=int) for k, v in groups.items()}
    
    raise ValueError(f"Unknown granularity: {granularity}")


def pairwise_comparison(singles_pred: Dict[str, np.ndarray], lbl: np.ndarray, 
                       ref_meta: dict) -> dict:
    """Pairwise McNemar test at 4 granularities."""
    results = {"all": {}, "subject": {}, "song": {}, "subject_song": {}}
    keys = list(singles_pred.keys())
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            pa, pb = singles_pred[a], singles_pred[b]
            
            # All samples
            results["all"][f"{a}|{b}"] = mcnemar_with_metrics(pa, pb, lbl)
            
            # By granularity
            for gran in ["subject", "song", "subject_song"]:
                groups = group_indices(ref_meta, gran)
                pair_results = {}
                for k, idx in groups.items():
                    if len(idx) == 0:
                        continue
                    pair_results[k] = mcnemar_with_metrics(pa[idx], pb[idx], lbl[idx])
                results[gran][f"{a}|{b}"] = pair_results
    
    return results


# ============================================================================
# Ensemble
# ============================================================================
def equal_weight_ensemble(meta_dicts: Dict[str, dict], members: List[str]) -> dict:
    """Equal-weight ensemble prediction."""
    base = meta_dicts[members[0]]
    
    # Average probabilities
    P = F.softmax(base["cls_logits"], dim=-1).clone()
    for m in members[1:]:
        P = P + F.softmax(meta_dicts[m]["cls_logits"], dim=-1)
    P = P / len(members)
    
    pred = P.argmax(dim=1).numpy()
    lbl = base["cls_lbl"].numpy()
    
    return {"pred": pred, "lbl": lbl}


def generate_ensemble_specs(model_keys: List[str], sizes: List[int]) -> List[Tuple[str, List[str]]]:
    """Generate ensemble specifications (name, members)."""
    ensembles = []
    for size in sizes:
        for combo in combinations(model_keys, size):
            name = "+".join(combo)
            ensembles.append((name, list(combo)))
    return ensembles


# ============================================================================
# Main evaluation
# ============================================================================
def evaluate(args, ckpts: Dict[str, str], out_dir: Path, logger):
    """Main evaluation pipeline."""
    # Build dataloader
    ds, loader = build_dataloader(args)
    
    # 1) Load or infer all single models
    logger.info("=== Loading/Inferring Single Models ===")
    meta = {}
    for model_key, ckpt_path in ckpts.items():
        meta[model_key] = load_or_infer(model_key, ckpt_path, args, out_dir, 
                                        ds, loader, logger)
    
    # 2) Single model accuracies
    singles_pred = {k: v["cls_logits"].argmax(dim=1).numpy() for k, v in meta.items()}
    lbl = meta[list(ckpts.keys())[0]]["cls_lbl"].numpy()
    singles_acc = {k: accuracy(p, lbl) for k, p in singles_pred.items()}
    
    logger.info("=== Single Model Accuracies ===")
    for k, acc in singles_acc.items():
        logger.info(f"{k}: {acc:.4f}")
    
    # 3) Pairwise comparison (4 granularities)
    logger.info("=== Pairwise Comparison (Single vs Single) ===")
    single_vs_single = pairwise_comparison(singles_pred, lbl, meta[list(ckpts.keys())[0]])
    
    # 4) Generate ensembles (2, 3, 4 models)
    model_keys = list(ckpts.keys())
    ensemble_specs = generate_ensemble_specs(model_keys, sizes=[2, 3, 4])
    
    logger.info(f"=== Evaluating {len(ensemble_specs)} Ensembles ===")
    ens_preds = {}
    ens_acc = {}
    for name, members in ensemble_specs:
        ens = equal_weight_ensemble(meta, members)
        ens_preds[name] = ens["pred"]
        ens_acc[name] = accuracy(ens["pred"], ens["lbl"])
        logger.info(f"{name}: {ens_acc[name]:.4f}")
    
    # 5) Ensemble vs Single
    logger.info("=== Ensemble vs Single ===")
    ens_vs_single = {}
    for e_name, e_pred in ens_preds.items():
        ens_vs_single[e_name] = {}
        for s_name, s_pred in singles_pred.items():
            ens_vs_single[e_name][s_name] = mcnemar_with_metrics(e_pred, s_pred, lbl)
    
    # 6) Ensemble vs Ensemble
    logger.info("=== Ensemble vs Ensemble ===")
    ens_names = list(ens_preds.keys())
    ens_vs_ens = {}
    for i, a in enumerate(ens_names):
        ens_vs_ens[a] = {}
        for b in ens_names[i + 1:]:
            ens_vs_ens[a][b] = mcnemar_with_metrics(ens_preds[a], ens_preds[b], lbl)
    
    return {
        "checkpoint_paths": ckpts,
        "single_accuracies": singles_acc,
        "single_vs_single": single_vs_single,
        "ensemble_accuracies": ens_acc,
        "ensemble_vs_single": ens_vs_single,
        "ensemble_vs_ensemble": ens_vs_ens
    }


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_arguments()
    out_dir = Path(args.out_dir)
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = out_dir / "log" / timestamp
    logger = setup_logger(log_dir)
    
    logger.info("=== PredANN+ Evaluation with McNemar Testing ===")
    
    # Build checkpoint map
    ckpts = build_ckpt_map(args)
    logger.info(f"Found {len(ckpts)} model checkpoints")
    
    if len(ckpts) == 0:
        logger.error("No checkpoints found. Check --ckpt_dir and seed arguments.")
        return
    
    # Run evaluation
    results = evaluate(args, ckpts, out_dir, logger)
    
    # Save results
    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    out_path = metrics_dir / f"evaluation_{timestamp}.json"
    
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved: {out_path}")


if __name__ == "__main__":
    main()
