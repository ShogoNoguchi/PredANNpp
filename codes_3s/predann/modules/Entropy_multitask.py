# coding: utf-8
"""
Multitask mode for joint Song ID and Entropy classification.
"""

import logging
import pandas as pd
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import timm
from itertools import chain

from predann.models import ms20_modeling_preEMenc


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
    def __init__(self, preprocess_dataset, args):
        super().__init__()
        self.save_hyperparameters(args)

        self.emenc = timm.create_model(
            "ms20_comp1_pretrain_ed_2layer_512",
            pretrained=False,
        )

        self.train_log_df = pd.DataFrame(columns=["Loss/train", "Accuracy/train_eeg"])
        self.valid_log_df = pd.DataFrame(columns=["Loss/valid", "Accuracy/valid_eeg"])

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
        eeg, label, ent_id, ent_raw = batch
        eeg = eeg.view(eeg.size(0), 128, 3, 125)
        entropy_logit, cls_token, mask_pos = self.emenc(eeg, Surp_or_Entropy_raw=ent_raw)
        return entropy_logit, cls_token, mask_pos

    def training_step(self, batch, _):
        eeg, label, ent_id, ent_raw = batch
        entropy_logit, cls_token, mask_pos = self.forward(batch)

        masked_logits = entropy_logit[mask_pos]
        masked_targets = ent_id[mask_pos]
        entropy_ce_loss = self.ce_loss(masked_logits, masked_targets)

        song_id_logit = self.projector1(self.norm(cls_token))
        song_id_ce_loss = self.ce_loss(song_id_logit, label)

        total_loss = 0.1 * entropy_ce_loss + 1.0 * song_id_ce_loss
        song_id_acc = (song_id_logit.argmax(dim=1) == label).float().mean()

        self.log("Entropy_CE_loss/train", entropy_ce_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/train", song_id_ce_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/train_total", total_loss, on_epoch=True, prog_bar=True)
        self.log("Accuracy/train_eeg", song_id_acc, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, _):
        eeg, label, ent_id, ent_raw = batch
        entropy_logit, cls_token, mask_pos = self.forward(batch)

        masked_logits = entropy_logit[mask_pos]
        masked_targets = ent_id[mask_pos]
        entropy_ce_loss = self.ce_loss(masked_logits, masked_targets)

        song_id_logit = self.projector1(self.norm(cls_token))
        song_id_ce_loss = self.ce_loss(song_id_logit, label)

        total_loss = 0.1 * entropy_ce_loss + 1.0 * song_id_ce_loss
        song_id_acc = (song_id_logit.argmax(dim=1) == label).float().mean()

        self.log("Entropy_CE_loss/valid", entropy_ce_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/valid", song_id_ce_loss, on_epoch=True, prog_bar=False)
        self.log("Loss/valid_total", total_loss, on_epoch=True, prog_bar=True)
        self.log("Accuracy/valid_eeg", song_id_acc, on_epoch=True, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            chain(self.emenc.parameters(), self.norm.parameters(), self.projector1.parameters()),
            lr=self.hparams.learning_rate,
        )
        return {"optimizer": optimizer}