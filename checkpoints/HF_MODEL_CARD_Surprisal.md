---
language: en
license: cc-by-nc-4.0
tags:
- eeg
- music
- representation-learning
- pytorch-lightning
- transformer
---

# PredANN++ (Surprisal, ctx16) — Encoder-only checkpoint

## Model description

This model is a finetuned **EEG encoder** for **song identification from 3-second EEG segments** (10-class classification) on the NMED‑T dataset.

The encoder is trained with multitask pretraining (masked prediction) using **MusicGen Surprisal** features, followed by finetuning with cross-entropy on song ID.

- Input: EEG (128 channels, 125 Hz, 3 seconds)
- Output: Song ID logits (10 classes)

## Intended use

- Research use for EEG-based music recognition
- Comparing the effect of predictive-information features (Surprisal) vs acoustic features

## Not intended use

- Medical diagnosis
- Any clinical decision making
- Commercial usage without verifying upstream non-commercial licenses

## Training data

- NMED‑T (Naturalistic Music EEG Dataset – Tempo), 10 songs, 20 subjects, trial=1.

## Training procedure (high-level)

1. Multitask pretraining: encoder-decoder masked prediction of Surprisal tokens (50% masking)
2. Finetuning: encoder-only training for Song ID classification

## Evaluation

The repository contains an evaluation script:
- `codes_3s/analysis/evaluate.py`

## License and upstream dependencies (IMPORTANT)

This checkpoint is trained using features computed with:
- MusicGen model weights (CC-BY-NC 4.0) :contentReference[oaicite:14]{index=14}

Therefore, this checkpoint is distributed as **CC-BY-NC 4.0**.

## Citation

If you use this model, please cite:
- The PredANN++ paper (to be added upon publication)
- NMED‑T dataset paper
- Audiocraft / MusicGen

## Acknowledgements

We borrow and adapt code from multiple repositories. Please see:
- `THIRD_PARTY_NOTICES.md` in the GitHub repository.