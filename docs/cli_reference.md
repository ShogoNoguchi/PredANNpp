# CLI Reference (PredANN++)

This document describes the public command-line interface (CLI) of the PredANN++ repository.

**Entry points**
- Training: `codes_3s/main_3s.py`
- Evaluation (single + ensemble + McNemar): `codes_3s/analysis/evaluate.py`

> This file documents only arguments that are actually registered by the public scripts.
> - For training (`main_3s.py`): all keys loaded from `codes_3s/config/config.yaml` plus flags explicitly added in `codes_3s/main_3s.py`
> - For evaluation (`evaluate.py`): all keys loaded from `codes_3s/config/config.yaml` plus evaluation-specific flags explicitly added in `codes_3s/analysis/evaluate.py`
>
> Undeclared internal references and stale documentation entries are intentionally excluded.

---

## Modes (Training)

`--mode` chooses the training objective:

- `Finetune`
  - Song ID classification using the encoder-only model.
  - If `--pretrain_ckpt_path` is `None` or `none`, training starts from scratch (**Fullscratch**).
- `MuQMultitask`
  - Song ID classification + 50% masked prediction of MuQ tokens (40 ms resolution).
- `SurpMultitask`
  - Song ID classification + 50% masked prediction of Surprisal tokens (20 ms resolution).
- `EntropyMultitask`
  - Song ID classification + 50% masked prediction of Entropy tokens (20 ms resolution).

**Important**
- The former `Pretrain` mode is not part of the public CLI.
- Use `Finetune` for both training from scratch and checkpoint-based fine-tuning.

---

## Dataset Path

To run successfully, provide the local NMED-T dataset directory.

Recommended:
- Pass `--dataset_dir <NMEDT_BASE_DIR>`.

Alternative:
- Edit `_base_dir` in `codes_3s/predann/datasets/preprocessing_eegmusic_dataset_3s.py`.

---

## Full Argument List (Training: `main_3s.py`)

### Infra / runtime

- `--gpus` (int): number of GPUs.
- `--accelerator` (str): accelerator string forwarded to the Lightning trainer (for example `dp`, `ddp`, `gpu`, `cpu`).
- `--workers` (int): number of DataLoader workers for training and validation.
- `--dataset_dir` (str): path to `<NMEDT_BASE_DIR>`.

### Training control

- `--seed` (int): global random seed.
- `--batch_size` (int): batch size.
- `--max_epochs` (int): maximum number of training epochs.
- `--eval_only` (int): if `1`, skip training and run validation only.
- `--training_date` (str): run name used for checkpoint and logging directories.

### Data / EEG

- `--dataset` (str): dataset loader name.
- `--eeg_normalization` (str): EEG normalization mode.
- `--eeg_sample_rate` (int): EEG sampling rate in Hz.
- `--eeg_length` (int): EEG clip length in samples.
- `--clamp_value` (int): clamp value used by `MetaAI` normalization.
- `--split_seed` (int): random seed used for the train/test split.
- `--class_song_id` (str): list-like string of target song IDs.
- `--shifting_time` (int): alignment shift parameter passed to the dataset loader.

### Sliding window

- `--window_size` (int): sliding-window size for `SW_*` subsets.
- `--stride` (int): sliding-window stride for `SW_*` subsets.
- `--start_position` (int): start-position offset inside the extracted window.

### Augmentation

- `--openmiir_augmentation` (str): legacy-named EEG augmentation selector.
  - `no_augmentation`: disable augmentation.
  - `gain`: apply random gain augmentation.
  - The gaussian-noise variants are not documented as supported public CLI modes in this release because their amplitude controls are not exposed as public CLI arguments.

### Optimization / compatibility fields

- `--optimizer` (str): optimizer field loaded from config for compatibility.
- `--learning_rate` (float): learning rate.
- `--supervised` (int): legacy config field retained for compatibility.
- `--train_test_splitting` (str): legacy config field retained for compatibility.

### Mode-related

- `--mode` (str): training mode (`Finetune`, `MuQMultitask`, `SurpMultitask`, `EntropyMultitask`).
- `--pretrain_ckpt_path` (str): checkpoint path used to initialize the encoder in `Finetune`.
- `--finetune_use_cls_token` (int): `1` uses the CLS token and `0` uses mean pooling.
- `--resume_from_checkpoint` (str): Lightning checkpoint path for resuming training.
- `--logger_version` (int): fixed TensorBoard version directory.
- `--accumulate_grad_batches` (int): gradient-accumulation steps.

### Experimental newMF

- `--use_new_mf` (int): if `1`, enable experimental 0.1 s-stride Surprisal/Entropy features for `SurpMultitask` and `EntropyMultitask`.
- `--new_mf_context_win` (int): context window in seconds for experimental newMF features (`8`, `16`, or `32`).

---

## Evaluation CLI (`evaluate.py`)

`evaluate.py` loads defaults from `codes_3s/config/config.yaml`, then adds the evaluation-specific public arguments below.

### Required I/O

- `--ckpt_dir` (str, required): directory containing checkpoint folders.
- `--out_dir` (str, required): output directory for cached logits and JSON metrics.

### Model selection

- `--fullscratch_seeds` (str): comma-separated seed list for Fullscratch models.
- `--multitask_seeds` (str): comma-separated seed list for Multitask→Finetune models.

### Execution mode

- `--mode` (str, required): evaluation execution mode.
  - `checkpoint`: run inference if cache is missing, otherwise load cache.
  - `offline`: use cache only.

### Evaluation runtime

- `--num_workers` (int): number of DataLoader workers used for evaluation.
- `--device` (str): evaluation device (`cpu` or `cuda`).
