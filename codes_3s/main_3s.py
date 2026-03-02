#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main_3s.py

Publication-scope training pipeline: Pretrain + Multitask + Finetune modes.

Modes:
- Pretrain: EEG encoder-only from scratch
- MuQMultitask: 50% masked multitask (JEPA + MuQ prediction)
- SurpMultitask: 50% masked multitask (JEPA + Surprisal prediction)
- EntropyMultitask: 50% masked multitask (JEPA + Entropy prediction)
- Finetune: Fine-tune encoder pretrained via multitask learning
"""
import argparse
import datetime
import random
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from audiomentations import AddGaussianNoise, Gain

from predann.datasets import get_dataset
from predann.utils import yaml_config_hook

# Paper scope modules (4 modes only)
from predann.modules.EM_finetune import TransformerEEGEncoder
from predann.modules.Surprisal_multitask import JEPA_Multitask as SurpMultitask
from predann.modules.MuQ_multitask import JEPA_Multitask as MuQMultitask
from predann.modules.Entropy_multitask import JEPA_Multitask as EntropyMultitask


def build_trainer(args, log_name: str, ckpt_dir: str):
    """
    Instantiate PyTorch Lightning Trainer with checkpoint and logging configuration.
    
    Monitors validation EEG accuracy; saves top-1 checkpoint and last epoch.
    """
    checkpoint_callback = ModelCheckpoint(
        monitor="Accuracy/valid_eeg",
        mode="max",
        save_top_k=1,
        save_last=True,
        dirpath=ckpt_dir,
        filename="best-{epoch:04d}-{Accuracy_valid_eeg:.4f}",
        save_on_train_epoch_end=False,
    )

    logger = TensorBoardLogger(
        save_dir="runs",
        name=log_name,
        version=args.logger_version,
    )

    trainer = Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=args.resume_from_checkpoint,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=args.max_epochs,
        deterministic=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=12,
        accelerator=args.accelerator,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[checkpoint_callback],
    )
    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune / Multitask")

    config_path = Path(__file__).parent / "config" / "config.yaml"
    config = yaml_config_hook(str(config_path))
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    parser.add_argument(
        "--mode",
        choices=["Finetune", "MuQMultitask", "SurpMultitask", "EntropyMultitask"],
        default="Finetune",
        help= "training mode. "
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="path to checkpoint for resuming training",
    )
    parser.add_argument(
        "--logger_version",
        type=int,
        default=None,
        help="fixed TensorBoard version (e.g., 0 appends to runs/<name>/version_0)",
    )

    parser.add_argument(
        "--pretrain_ckpt_path",
        type=str,
        default=None,
        help="pretrained checkpoint for fine-tuning initialization",
    )
    parser.add_argument(
        "--finetune_use_cls_token",
        type=int,
        default=1,
        choices=[0, 1],
        help="use CLS token in fine-tuning: 1=CLS, 0=mean pooling",
    )

    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="gradient accumulation steps",
    )
    parser.add_argument(
        "--start_position",
        type=int,
        default=0,
        help="start position for sliding window",
    )

    parser.add_argument(
        "--use_new_mf",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "0: traditional Surprisal/Entropy (30s chunk, discretized, default), "
            "1: new Surprisal/Entropy with 0.1s stride (experimental)"
        ),
    )
    parser.add_argument(
        "--new_mf_context_win",
        type=int,
        default=8,
        choices=[8, 16, 32],
        help=(
            "context window in seconds for new MuF features: "
            "8=SurpEnt0.1stride, 16=SurpEnt0.1stride_ctx16, 32=SurpEnt0.1stride_ctx32"
        ),
    )

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    train_transform = {}
    if args.openmiir_augmentation == "gaussiannoise":
        train_transform = [
            AddGaussianNoise(
                min_amplitude=args.min_amplitude,
                max_amplitude=args.max_amplitude,
                p=0.5,
            ),
        ]
        print("augmentation is gaussiannoise")
    elif args.openmiir_augmentation == "gain":
        train_transform = [Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)]
        print("augmentation is gain")
    elif args.openmiir_augmentation == "gaussiannoise+gain":
        train_transform = [
            AddGaussianNoise(
                min_amplitude=args.min_amplitude,
                max_amplitude=args.max_amplitude,
                p=0.5,
            ),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5),
        ]
        print("augmentation is gaussiannoise+gain")
    else:
        print("no augmentation")

    train_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="SW_train", download=False
    )
    train_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    train_dataset.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    train_dataset.set_other_parameters(
        args.eeg_length,
        args.audio_clip_length,
        args.split_seed,
        args.class_song_id,
        args.shifting_time,
        args.start_position,
    )
    random.seed(args.seed)
    train_random_numbers = [
        random.randint(0, 125 * 30 - 375 - 1) for _ in range(1200)
    ]
    train_dataset.set_random_numbers(train_random_numbers)
    train_dataset.set_mode(args.mode)

    if hasattr(train_dataset, "set_new_mf_flags"):
        use_new_surp = (args.use_new_mf == 1) and (args.mode == "SurpMultitask")
        use_new_ent = (args.use_new_mf == 1) and (args.mode == "EntropyMultitask")
        train_dataset.set_new_mf_flags(use_new_surp=use_new_surp, use_new_ent=use_new_ent)

    if hasattr(train_dataset, "set_new_mf_context_win"):
        train_dataset.set_new_mf_context_win(args.new_mf_context_win)

    if args.openmiir_augmentation != "no_augmentation":
        train_dataset.set_transform(train_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    valid_dataset = get_dataset(
        args.dataset, args.dataset_dir, subset="SW_valid", download=False
    )
    valid_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    valid_dataset.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    valid_dataset.set_other_parameters(
        args.eeg_length,
        args.audio_clip_length,
        args.split_seed,
        args.class_song_id,
        args.shifting_time,
        args.start_position,
    )
    random.seed(args.seed)
    valid_random_numbers = [
        random.randint(0, args.window_size - 375 - 1) for _ in range(1200)
    ]
    valid_dataset.set_random_numbers(valid_random_numbers)
    valid_dataset.set_mode(args.mode)

    if hasattr(valid_dataset, "set_new_mf_flags"):
        use_new_surp = (args.use_new_mf == 1) and (args.mode == "SurpMultitask")
        use_new_ent = (args.use_new_mf == 1) and (args.mode == "EntropyMultitask")
        valid_dataset.set_new_mf_flags(use_new_surp=use_new_surp, use_new_ent=use_new_ent)

    if hasattr(valid_dataset, "set_new_mf_context_win"):
        valid_dataset.set_new_mf_context_win(args.new_mf_context_win)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )

    print(f"Size of train dataset: {len(train_dataset)}")
    print(f"Size of valid dataset: {len(valid_dataset)}")

    if args.mode == "Finetune":
        module = TransformerEEGEncoder(train_dataset, args)
        if getattr(args, "training_date", None):
            ckpt_dir = f"best_checkpoints/{args.training_date}/SongAcc/"
            log_name = args.training_date
        else:
            ckpt_dir = "best_checkpoints/finetune/SongAcc/"
            log_name = "finetune"

    elif args.mode == "MuQMultitask":
        module = MuQMultitask(train_dataset, args)
        if getattr(args, "training_date", None):
            ckpt_dir = f"best_checkpoints/{args.training_date}/SongAcc/"
            log_name = args.training_date
        else:
            ckpt_dir = "best_checkpoints/50perMask/multitask/MuQMultitask/SongAcc/"
            log_name = "50per_MuQMultitask"

    elif args.mode == "SurpMultitask":
        module = SurpMultitask(train_dataset, args)
        if getattr(args, "training_date", None):
            ckpt_dir = f"best_checkpoints/{args.training_date}/SongAcc/"
            log_name = args.training_date
        else:
            ckpt_dir = "best_checkpoints/50perMask/multitask/SurpMultitask/SongAcc/"
            log_name = "50per_SurpMultitask"

    elif args.mode == "EntropyMultitask":
        module = EntropyMultitask(train_dataset, args)
        if getattr(args, "training_date", None):
            ckpt_dir = f"best_checkpoints/{args.training_date}/SongAcc/"
            log_name = args.training_date
        else:
            ckpt_dir = "best_checkpoints/50perMask/multitask/EntropyMultitask/SongAcc/"
            log_name = "50per_EntropyMultitask"

    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    trainer = build_trainer(args, log_name, ckpt_dir)

    print("[[[ START ]]]", datetime.datetime.now())
    if args.eval_only:
        print("[[ EVAL ONLY MODE ]]")
        trainer.validate(module, dataloaders=valid_loader)
    else:
        trainer.fit(module, train_loader, valid_loader)
    print("[[[ FINISH ]]]", datetime.datetime.now())