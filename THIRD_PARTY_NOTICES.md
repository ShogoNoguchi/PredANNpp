# Third-Party Notices for PredANN++

Last verified: 2026-02-05

This repository (ShogoNoguchi/MusicFeaturePred_Surp_Ent_MuQ, PredANN+ branch) is released under **CC BY-SA 4.0** unless otherwise noted in a file header.

We gratefully acknowledge and respect the following third-party resources.

---

## PredANN (Base code structure, CC BY-SA 4.0)

This repository is developed based on the code structure and implementation ideas of **PredANN** (CC BY-SA 4.0).
We have adapted and extended multiple components including directory layout, training scripts, and evaluation utilities.

---

## LaBraM (Selected encoder components, MIT License)

In `codes_3s/predann/models/modeling_fineEMenc.py`, the following components are adapted from LaBraM's `modeling_finetune.py`
under the MIT License (as stated by the upstream project):

- `TemporalConv`
- `drop_path`
- `DropPath`
- `Mlp`
- `Attention`
- `Block`

We added explicit attribution comments in that file and preserved the MIT license text snippet in the header section.

---

## Audiocraft / MusicGen (Surprisal & Entropy extraction)

We use **Audiocraft** for MusicGen inference.

- Audiocraft code license: **MIT** :contentReference[oaicite:2]{index=2}  
- MusicGen model weights license: **CC-BY-NC 4.0** :contentReference[oaicite:3]{index=3}  

IMPORTANT:
- Because Surprisal/Entropy are computed using MusicGen model weights (CC-BY-NC 4.0),
  downstream artifacts trained on these features may inherit non-commercial restrictions.
  Please verify your intended usage and comply with all applicable licenses.

Repositories / model pages:
- Audiocraft: https://github.com/facebookresearch/audiocraft :contentReference[oaicite:4]{index=4}
- musicgen-large model card: https://huggingface.co/facebook/musicgen-large :contentReference[oaicite:5]{index=5}

---

## MuQ (Acoustic embedding extraction)

We optionally use MuQ for extracting acoustic embeddings.

OpenMuQ checkpoint license information (Hugging Face model card):
- Code: **MIT**
- Model weights: **CC-BY-NC 4.0** :contentReference[oaicite:6]{index=6}

Repository / model pages:
- MuQ repository: https://github.com/tencent-ailab/MuQ/tree/main
- OpenMuQ checkpoint: https://huggingface.co/OpenMuQ/MuQ-large-msd-iter :contentReference[oaicite:7]{index=7}

---

## NMED-T dataset (EEG + audio)

This work uses the Naturalistic Music EEG Dataset – Tempo (NMED‑T).
This repository does not redistribute NMED‑T data files.
Please obtain the dataset from the official distribution channel and follow its license terms.

---

## Additional third-party Python packages

This project uses common scientific Python packages (PyTorch, NumPy, SciPy, pandas, scikit-learn, statsmodels, etc.).
Please refer to each package's license for details.
