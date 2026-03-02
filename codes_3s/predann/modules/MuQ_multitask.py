
# coding: utf-8
"""
Multitask mode for joint Song ID and MuQ discretized embedding prediction.
Song ID classification is used in conjunction to improve the prediction accuracy of MuQ prediction.
"""

import os, json, logging, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from pytorch_lightning import LightningModule
import timm      
from predann.models import ms40_modeling_preEMenc

from pathlib import Path
import sys, logging
import math
from itertools import chain

def _setup_logger():
    logger = logging.getLogger("dataloader_debug")
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler("dataloader_debug.log")
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False  
    return logger

debug_logger = _setup_logger()

class JEPA_Multitask(LightningModule):
    """
    Joint MuQ and Song ID classification from 3-second EEG data.
    
    Input:
        eeg: [B, 128, 3, 125] (128 channels, 3 seconds, 125 Hz)
    
    Output:
        Song ID logits: (B, 10) for 10-song classification
        MuQ logits: (B, 75, 128) for 75 tokens at 40ms intervals, 128 discrete classes
    """

    def __init__(self, preprocess_dataset, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.emenc = timm.create_model(
            "ms40_comp1_pretrain_ed_2layer_512",
            pretrained=False,
        )

        self.train_log_df = pd.DataFrame(columns=["Loss/train", "Accuracy/train_eeg"])
        self.valid_log_df = pd.DataFrame(columns=["Loss/valid", "Accuracy/valid_eeg"])

        self.last_epoch_train_embeddings = []
        self.last_epoch_train_labels = []
        self.last_epoch_valid_embeddings = []
        self.last_epoch_valid_labels = []
        self.norm = nn.LayerNorm(512, eps=1e-6)
        self.preprocess_dataset = preprocess_dataset
        out_dim = preprocess_dataset.labels()

        self.projector1 = nn.Sequential(
            nn.Linear(512, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim, bias=False),
        )

        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        """
        Forward pass for MuQ classification.
        
        Args:
            batch: Tuple of (eeg, label, muq_id, muq_raw)
                eeg: (B, 128, 3, 125)
                label: (B,) Song ID labels
                muq_id: (B, 75) Discretized MuQ embeddings
                muq_raw: (B, 75, 1024) Continuous MuQ embeddings
        
        Returns:
            MuQ_logit: (B, 75, 128) Logits for each token
            cls_token: (B, 512) CLS token representation
            mask_pos: (B, 75) Boolean mask indicating masked positions
        """
        eeg, label, muq_id, muq_raw = batch
        eeg = eeg.view(eeg.size(0), 128, 3, 125)
        MuQ_logit, cls_token, mask_pos = self.emenc(eeg, muq_raw=muq_raw)
        return MuQ_logit, cls_token, mask_pos
    def training_step(self, batch, _):
        eeg, label, muq_id, muq_raw = batch
        MuQ_logit, cls_token, mask_pos = self.forward(batch)

        # Masked positions only: avoid predicting unmasked tokens
        masked_logits = MuQ_logit[mask_pos]
        masked_targets = muq_id[mask_pos]
        MuQ_CE_loss = self.ce_loss(masked_logits, masked_targets)

        Song_id_logit = self.projector1(self.norm(cls_token))
        Song_id_CE_loss = self.ce_loss(Song_id_logit, label)

        Total_loss = 0.1 * MuQ_CE_loss + 1.0 * Song_id_CE_loss
        Song_id_Acc = (Song_id_logit.argmax(dim=1) == label).float().mean()

        self.log("MuQ_CE_loss/train", MuQ_CE_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/train", Song_id_CE_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/train_total", Total_loss, on_epoch=True, prog_bar=True)
        self.log("Accuracy/train_eeg", Song_id_Acc, on_epoch=True, prog_bar=True)
  

        return Total_loss

    def validation_step(self, batch, _):
        eeg, label, muq_id, muq_raw = batch
        MuQ_logit, cls_token, mask_pos = self.forward(batch)

        # Masked positions only: avoid predicting unmasked tokens
        masked_logits = MuQ_logit[mask_pos]
        masked_targets = muq_id[mask_pos]
        MuQ_CE_loss = self.ce_loss(masked_logits, masked_targets)

        Song_id_logit = self.projector1(self.norm(cls_token))
        Song_id_CE_loss = self.ce_loss(Song_id_logit, label)

        Total_loss = 0.1 * MuQ_CE_loss + 1.0 * Song_id_CE_loss
        Song_id_Acc = (Song_id_logit.argmax(dim=1) == label).float().mean()

        self.log("MuQ_CE_loss/valid", MuQ_CE_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/valid", Song_id_CE_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/valid_total", Total_loss, on_epoch=True, prog_bar=True)
        self.log("Accuracy/valid_eeg", Song_id_Acc, on_epoch=True, prog_bar=True)
  

        return Total_loss



    def on_validation_start(self):
        self.eval()

    def Kfold_log(self):
        return self.train_log_df, self.valid_log_df

    def save_checkpoint(self, filepath: str):
        torch.save(
            {
                "module_state_dict": self.state_dict(),
                "emenc_state_dict":  self.emenc.state_dict(),
                "optimizer_state_dict": self.trainer.optimizers[0].state_dict(),
            },
            filepath,
        )

    def load_checkpoint(self, filepath: str):
        ckpt = torch.load(filepath, map_location="cpu")
        self.load_state_dict(ckpt["module_state_dict"])
        self.emenc.load_state_dict(ckpt["emenc_state_dict"])
        self.norm.load_state_dict(ckpt["norm_state_dict"])
        self.projector1.load_state_dict(ckpt["projector1_state_dict"])

        opt = self.configure_optimizers()["optimizer"]
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        return opt

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(
                self.emenc.parameters(),
                self.norm.parameters(),
                self.projector1.parameters()
            ),
            lr=self.hparams.learning_rate
        )
        return {"optimizer": optimizer}

