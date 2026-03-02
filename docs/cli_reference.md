# CLI Reference (PredANN++)

This document describes the command-line interface (CLI) of the public PredANN+ repository.

**Entry points**
- Training: `codes_3s/main_3s.py`
- Evaluation (single + ensemble + McNemar): `codes_3s/analysis/evaluate.py`

> Note: This repository intentionally keeps the CLI small. Some additional arguments may be
> available via PyTorch Lightning, but the arguments below are the officially supported ones.

---

## Modes (Training)

`--mode` chooses the training objective:

- `Finetune`
  - Song ID classification using the encoder-only model.
  - If `--pretrain_ckpt_path` is `None` / `none`, training starts from scratch (**Fullscratch**).
- `MuQMultitask`
  - Song ID classification + 50% masked prediction of MuQ tokens (40 ms resolution).
- `SurpMultitask`
  - Song ID classification + 50% masked prediction of Surprisal tokens (20 ms resolution).
- `EntropyMultitask`
  - Song ID classification + 50% masked prediction of Entropy tokens (20 ms resolution).

**Important**
- The former `Pretrain` mode has been removed.
  Use `Finetune` for both “from scratch” and “fine-tuning from a checkpoint”.

---

## Dataset Path

To run successfully, you must provide the local NMED-T dataset directory.

Recommended:
- Pass `--dataset_dir <NMEDT_BASE_DIR>`.

Alternative:
- Edit `_base_dir` in `codes_3s/predann/datasets/preprocessing_eegmusic_dataset_3s.py`.

---

## Full Argument List (main_3s.py)

This list contains:
- all keys from `codes_3s/config/config.yaml`
- additional CLI flags explicitly defined in `codes_3s/main_3s.py`

Each option is described in one short line.

### Infra / runtime

- `--gpus` (int): number of GPUs (Lightning Trainer flag).
- `--accelerator` (str): Lightning accelerator (e.g., `dp`, `ddp`, `gpu`, `cpu`).
- `--workers` (int): DataLoader workers for training/validation.
- `--dataset_dir` (str): path to `<NMEDT_BASE_DIR>` (directory containing `audio/`, `DS_EEG_pkl/`, etc.).

### Training control

- `--seed` (int): global seed (reproducibility).
- `--batch_size` (int): batch size.
- `--max_epochs` (int): number of epochs to train.
- `--eval_only` (int): if 1, run validation only.
- `--train_only` (int): reserved flag (kept for compatibility).
- `--training_date` (str): run name used for checkpoint/log directories.

### Data / EEG

- `--dataset` (str): dataset name (default: `preprocessing_eegmusic`).
- `--eeg_type` (str): EEG type string (e.g., `raw`).
- `--eeg_normalization` (str): EEG normalization mode (e.g., `MetaAI`).
- `--clamp_value` (int): clamp value for `MetaAI` normalization.
- `--eeg_sample_rate` (int): EEG sampling rate (Hz).
- `--audio_sample_rate` (int): audio sampling rate (Hz).
- `--eeg_length` (int): EEG clip length in samples (default 375 = 3s at 125 Hz).
- `--audio_clip_length` (float): audio context length in seconds (used for cropping margins).
- `--split_seed` (int): random seed used for train/test split.
- `--class_song_id` (str): list-like string of song IDs (e.g., `[21,22,...,36]`).
- `--shifting_time` (int): time shift parameter for alignment (in ms-like scale used by code).

### Sliding window

- `--window_size` (int): sliding window size for SW_* subsets.
- `--stride` (int): stride size for SW_* subsets.
- `--start_position` (int): start position offset inside the extracted window.

### Augmentation (EEG augmentation via audiomentations)

- `--openmiir_augmentation` (str): `no_augmentation`, `gaussiannoise`, `gain`, `gaussiannoise+gain`.
- `--max_amplitude` (float): max amplitude for gaussian noise.
- `--min_amplitude` (float): min amplitude for gaussian noise.

### Optimization (kept as config fields)

- `--optimizer` (str): optimizer name (currently kept for compatibility).
- `--learning_rate` (float): learning rate.
- `--weight_decay` (float): weight decay.
- `--alpha` (float): reserved weight parameter.
- `--supervised` (int): reserved flag.
- `--loss_function` (str): loss name string (kept for compatibility).
- `--detach_z_audio` (int): reserved flag.
- `--weight_r` (float): reserved weight parameter.
- `--weight_c` (float): reserved weight parameter.
- `--weight_predann` (float): reserved weight parameter.
- `--dim_reduction` (int): reserved flag.
- `--train_test_splitting` (str): reserved split mode string.

### Mode-related

- `--mode` (str): training mode (`Finetune`, `MuQMultitask`, `SurpMultitask`, `EntropyMultitask`).
- `--pretrain_ckpt_path` (str): checkpoint path for initializing encoder in `Finetune`.
- `--finetune_use_cls_token` (int): 1=use CLS token, 0=mean pooling.
- `--resume_from_checkpoint` (str): Lightning resume checkpoint.
- `--logger_version` (int): TensorBoard version folder number.
- `--accumulate_grad_batches` (int): gradient accumulation steps.

### Experimental newMF (Surprisal/Entropy with 0.1 s stride)

- `--use_new_mf` (int): 1 enables newMF for Surp/Entropy multitask, 0 uses 30s-chunk features.
- `--new_mf_context_win` (int): newMF context length (8/16/32 seconds).

---

## Evaluation CLI (evaluate.py)

Main arguments:

- `--ckpt_dir` (str, required): directory containing checkpoint folders.
- `--out_dir` (str, required): output directory for cached logits and JSON metrics.
- `--mode` (str, required): `checkpoint` (infer if cache missing) or `offline` (cache only).
- `--fullscratch_seeds` (str): comma-separated seeds for Fullscratch.
- `--multitask_seeds` (str): comma-separated seeds for Multitask→Finetune.
- `--num_workers` (int): DataLoader workers for evaluation.
- `--device` (str): `cpu` or `cuda`.
