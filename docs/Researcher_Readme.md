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

## 5) GitHub Pages local test (before push)

For the synchronized demo-track visualization in `docs/index.html`, always test via localhost (not `file://`).

```bash
cd docs
python3 -m http.server 8000
```

Open:

```text
http://localhost:8000/index.html
```

Checklist for the `Demo Track Sync Visualization (continuous timeline)` section:

- Source selector switches among 3 demo tracks
- Data load succeeds from `docs/assets/data/manifest.json` and per-track JSON files
- Audio loads from `docs/assets/audio/*.mp3`
- Feature toggle switches `Surprisal (Q128)` and `Entropy (Q128)`
- Playback/visualization upper bound is 240 sec, while shorter tracks end at natural duration
- Zoom and timeline interactions update the plotted time range continuously
- Red playback head moves smoothly during play
- Clicking or dragging on the graph seeks `audio.currentTime` correctly

## 6) Discretization method used for demo curves

Surprisal and Entropy continuous sequences were pooled across all songs (NMED-T 10 songs + demo 3 songs),
then feature-wise 128-bin equal-frequency quantile binning was applied.
Using the resulting bin boundaries, each frame value was converted to a discrete label in the range 0–127.

## 7) Demo audio licensing notes

The 3 demo audio tracks are extracted from MTG-Jamendo and are provided for non-commercial research/academic use.
Track-specific attributions and Creative Commons licenses are listed in `docs/audio_licenses.txt`.
For commercial use, contact Jamendo licensing team: `hello@jamendo.com`.

