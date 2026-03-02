# PredANN++ Checkpoints (Encoder-only)

Last updated: 2026-02-05

This folder provides **ready-to-use finetuned checkpoints** for quick evaluation and demo.

Available files:
- `PredANNpp_SongAcc_Entropy_ctx16_Pretrain10kEpoch_Finetune3.5kEpoch_seed42.ckpt`
- `PredANNpp_SongAcc_Surprisal_ctx16_Pretrain10kEpoch_Finetune3.5kEpoch_seed42.ckpt`

Both checkpoints are intended for **3-second EEG → Song ID (10-class)** inference and downstream fine-tuning.

> NOTE
> - These checkpoints contain an **encoder (and the Song-ID classifier head)**.
> - The multitask **decoder** is not used in finetuning and is not required for demo inference.

---

## Quick demo (recommended)

Use the repository-level Gradio demo:

```bash
pip install -r requirements.txt
pip install -r requirements_demo.txt

python demo.py --dataset_dir /path/to/NMED-T_dataset
```
## Reproducibility note (data is not included)
NMED‑T EEG/audio data is **not distributed** in this repository.

Please download the official NMED‑T dataset and place it as described in:
- `PredANN+/Readme.md`
- `PredANN+/scripts/data_prep/README.md`
## License for checkpoints (IMPORTANT)
These checkpoints are trained using features derived from:

- MusicGen model weights (CC-BY-NC 4.0) https://huggingface.co/facebook/musicgen-large 
- MuQ weights (CC-BY-NC 4.0) https://huggingface.co/OpenMuQ/MuQ-large-msd-iter 

Therefore, to be safe and compliant in public distribution, we treat these checkpoints as:

- **CC-BY-NC 4.0 (Non-Commercial)**

If you need commercial usage, please contact the authors and also verify upstream license terms.
## Model cards
Hugging Face model cards (copy-paste templates) are provided here:
- `HF_MODEL_CARD_Entropy.md`
- `HF_MODEL_CARD_Surprisal.md`
