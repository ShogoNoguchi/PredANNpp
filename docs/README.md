# Researcher README (PredANN++)

This folder contains researcher-oriented documentation that does not rely on `.sh` wrappers.

---

## 1) Dataset Path Configuration

You must point the code to your local NMED-T dataset directory.

Recommended:
- pass `--dataset_dir <NMEDT_BASE_DIR>` to `codes_3s/main_3s.py`

Alternative:
- edit `_base_dir` in:
  - `codes_3s/predann/datasets/preprocessing_eegmusic_dataset_3s.py`

Example:

```python
_base_dir = "/path/to/NMED-T_dataset"
```

## 2) Example Commands (Training)

All examples below assume you run commands from the repository root.

### 2.1 Fullscratch (train from scratch)

This uses Finetune mode with no checkpoint initialization:

```bash
nohup bash -c "
python codes_3s/main_3s.py \
  --mode Finetune \
  --dataset_dir /path/to/NMED-T_dataset \
  --gpus 1 \
  --workers 8 \
  --max_epochs 3500 \
  --batch_size 48 \
  --seed 42 \
  --logger_version 0 \
  --training_date Fullscratch_seed42
" > codes_3s/log/Fullscratch_seed42.log 2>&1 &
```

Checkpoints are saved into:
- codes_3s/best_checkpoints/Fullscratch_seed42/ (contains last.ckpt)

### 2.2 Multitask (Pretraining)
**MuQ Multitask** 

```bash
nohup bash -c "
python codes_3s/main_3s.py \
  --mode MuQMultitask \
  --dataset_dir /path/to/NMED-T_dataset \
  --gpus 1 \
  --workers 8 \
  --max_epochs 10000 \
  --batch_size 48 \
  --seed 42 \
  --logger_version 0 \
  --training_date MuQMultitask_seed42
" > codes_3s/log/MuQMultitask_seed42.log 2>&1 &
```

**Surprisal Multitask**

```
nohup bash -c "
python codes_3s/main_3s.py \
  --mode SurpMultitask \
  --dataset_dir /path/to/NMED-T_dataset \
  --gpus 1 \
  --workers 8 \
  --max_epochs 10000 \
  --batch_size 48 \
  --seed 42 \
  --logger_version 0 \
  --training_date SurpMultitask_seed42
" > codes_3s/log/SurpMultitask_seed42.log 2>&1 &
```

**Entropy Multitask**

```
nohup bash -c "
python codes_3s/main_3s.py \
  --mode EntropyMultitask \
  --dataset_dir /path/to/NMED-T_dataset \
  --gpus 1 \
  --workers 8 \
  --max_epochs 10000 \
  --batch_size 48 \
  --seed 42 \
  --logger_version 0 \
  --training_date EntropyMultitask_seed42
" > codes_3s/log/EntropyMultitask_seed42.log 2>&1 &
```

## Example Commands (Finetune from a Multitask Checkpoint)

Use [Finetune] mode and point [--pretrain_ckpt_path] to the multitask last.ckpt.

```bash
nohup bash -c "
python codes_3s/main_3s.py \
  --mode Finetune \
  --dataset_dir /path/to/NMED-T_dataset \
  --gpus 1 \
  --workers 8 \
  --max_epochs 3500 \
  --batch_size 48 \
  --seed 42 \
  --logger_version 0 \
  --pretrain_ckpt_path codes_3s/best_checkpoints/MuQMultitask_seed42/last.ckpt \
  --training_date MuQMultitask_Finetune_seed42
" > codes_3s/log/MuQMultitask_Finetune_seed42.log 2>&1 &
```

## 4) Evaluation (Single + Ensemble + McNemar)

Run:

```bash
python codes_3s/analysis/evaluate.py \
  --ckpt_dir codes_3s/best_checkpoints \
  --out_dir ./analysis_outputs \
  --fullscratch_seeds "42,1,2,3" \
  --multitask_seeds "42,1,2,3" \
  --mode checkpoint
```
Outputs:

- Cached logits: analysis_outputs/logits/...
- JSON metrics: analysis_outputs/metrics/evaluation_YYYYMMDD_HHMMSS.json

