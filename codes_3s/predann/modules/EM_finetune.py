"""
Music classification model for full training from scratch or fine-tuning.
Encoder-only version of the multi-task model.

License:
- CC-BY-SA 4.0 (repository license)
"""

import glob
import torch
from pytorch_lightning import LightningModule
import pandas as pd
import logging
import os
from itertools import chain

import timm
import torch.nn as nn

from predann.models import modeling_fineEMenc


def setup_logger():
    if os.path.exists("dataloader_debug.log"):
        os.remove("dataloader_debug.log")
    logger = logging.getLogger("dataloader_debug")
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("dataloader_debug.log")
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.propagate = False
    return logger


debug_logger = setup_logger()


class TransformerEEGEncoder(LightningModule):
    def __init__(self, preprocess_dataset, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.args = args
        # ===== IMPORTANT: make args robust across tools (evaluate.py / demo.py) =====
        self.pretrain_ckpt_path = getattr(args, "pretrain_ckpt_path", None)

        out_dim = preprocess_dataset.labels()

        # CLS token vs. pooling selector for architecture flexibility
        self.use_cls_token = True
        if hasattr(args, "finetune_use_cls_token"):
            try:
                self.use_cls_token = bool(int(getattr(args, "finetune_use_cls_token")))
            except Exception as e:
                print(f"[WARN] finetune_use_cls_token parse error: {e}. fallback to True")
                self.use_cls_token = True

        print(f"[INFO] Finetune use_cls_token = {self.use_cls_token}")
        debug_logger.debug(f"Finetune use_cls_token = {self.use_cls_token}")

        self.emenc = timm.create_model(
            "comp1_fineEEGenc_2layer_512",
            pretrained=False,
            use_cls_token=self.use_cls_token,
        )

        self.validation_end_values = []

        self.batch_accuracies = []
        self.label_accuracy_count = {label: {"correct": 0, "total": 0} for label in range(10)}
        self.subject_accuracy_count = {subject: {"correct": 0, "total": 0} for subject in range(24)}

        self.train_log_df = pd.DataFrame(columns=["Loss/train", "Accuracy/train_eeg"])
        self.valid_log_df = pd.DataFrame(columns=["Loss/valid", "Accuracy/valid_eeg"])

        self.last_epoch_train_embeddings = []
        self.last_epoch_train_labels = []
        self.last_epoch_valid_embeddings = []
        self.last_epoch_valid_labels = []

        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.preprocess_dataset = preprocess_dataset

        self.projector1 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim, bias=False),
        )

        self.ce_loss = nn.CrossEntropyLoss()

        # Load encoder weights from a pretrain checkpoint if provided
        self.load_emenc_from_pretrain(self.pretrain_ckpt_path)

    def forward(self, eeg):
        """
        Forward pass through the EEG encoder.

        Args:
            eeg: Tensor of shape [B, 128, 375]

        Returns:
            Tensor of shape [B, 512] (representation vector)
        """
        B, C, L = eeg.shape
        assert C == 128 and L == 375, f"Expected shape [B,128,375], got {eeg.shape}"
        eeg_3s = eeg.view(B, C, 3, 125)
        cls_tok_hid = self.emenc(eeg_3s)
        return cls_tok_hid

    def load_emenc_from_pretrain(self, ckpt_path):
        if not ckpt_path or str(ckpt_path).lower() == "none":
            print("[INFO] train from scratch")
            return

        latest_ckpt = ckpt_path if str(ckpt_path).endswith(".ckpt") else sorted(glob.glob(os.path.join(ckpt_path, "*.ckpt")))[-1]
        print(f"[INFO] load emenc from {latest_ckpt}")

        ckpt = torch.load(latest_ckpt, map_location="cpu")
        state_dict = ckpt["state_dict"]

        emenc_sd = {}
        for k, v in state_dict.items():
            if not k.startswith("emenc."):
                continue
            nk = k[len("emenc.") :]

            # Strict=False allows layers from pretraining to be missing in fine-tune model
            if (
                nk.startswith("decoder")
                or nk.startswith("proj_out")
                or nk == "mask_token"
                or nk.startswith("time40_emb")
                or nk.startswith("time20_emb")
                or nk.startswith("decoder_mask_norm")
                or nk.startswith("music_feat_proj")
            ):
                print(f"[DROP] {k}")
                continue

            if hasattr(self.emenc, "use_cls_token") and (not bool(getattr(self.emenc, "use_cls_token"))) and nk == "cls_token":
                print(f"[DROP] {k} (CLS disabled)")
                continue

            emenc_sd[nk] = v
            print(f"[LOAD] {k} -> {nk}")

        msg = self.emenc.load_state_dict(emenc_sd, strict=False)
        print("[INFO] strict load ok :", msg)

    def training_step(self, batch, _):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            eeg, label = batch
        else:
            eeg, label = batch[0], batch[1]

        cls_token = self.forward(eeg)
        song_id_logit = self.projector1(self.norm(cls_token))
        song_id_ce_loss = self.ce_loss(song_id_logit, label)
        song_id_acc = (song_id_logit.argmax(dim=1) == label).float().mean()

        self.log("Loss/train", song_id_ce_loss, on_epoch=True, prog_bar=False)
        self.log("Accuracy/train_eeg", song_id_acc, on_epoch=True, prog_bar=True)
        return song_id_ce_loss

    def validation_step(self, batch, _):
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            eeg, label = batch
        else:
            eeg, label = batch[0], batch[1]

        cls_token = self.forward(eeg)
        song_id_logit = self.projector1(self.norm(cls_token))
        song_id_ce_loss = self.ce_loss(song_id_logit, label)
        song_id_acc = (song_id_logit.argmax(dim=1) == label).float().mean()

        self.log("Loss/valid", song_id_ce_loss, on_epoch=True, prog_bar=False)
        self.log("Accuracy/valid_eeg", song_id_acc, on_epoch=True, prog_bar=True)
        return song_id_ce_loss

    def Kfold_log(self):
        return self.train_log_df, self.valid_log_df

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(
                self.emenc.parameters(),
                self.norm.parameters(),
                self.projector1.parameters(),
            ),
            lr=self.hparams.learning_rate,
        )
        return {"optimizer": optimizer}