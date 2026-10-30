"""Full encoder–decoder model."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadseq.tokenizer import PAD_ID

from .decoder import SeqDecoder
from .encoder import BRepEncoder


@dataclass
class ModelConfig:
    d_model: int = 384
    heads: int = 8
    graph_layers: int = 4
    enc_layers: int = 3
    dec_layers: int = 8
    dropout: float = 0.1
    max_len: int = 1024


class Solid2Param(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        self.encoder = BRepEncoder(cfg.d_model, cfg.graph_layers, cfg.enc_layers, cfg.heads, cfg.dropout)
        self.decoder = SeqDecoder(cfg.d_model, cfg.dec_layers, cfg.heads, cfg.dropout, cfg.max_len)

    def encode(self, batch):
        return self.encoder(batch["node_x"], batch["edge_x"], batch["edge_index"],
                            batch["node_batch"], batch["tokens"].shape[0])

    def forward(self, batch):
        mem, pad = self.encode(batch)
        return self.decoder(batch["tokens"][:, :-1], mem, pad)

    def loss(self, batch, label_smoothing: float = 0.0):
        logits = self(batch)
        target = batch["tokens"][:, 1:]
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1),
                               ignore_index=PAD_ID, label_smoothing=label_smoothing)
        with torch.no_grad():
            valid = target != PAD_ID
            pred = logits.argmax(-1)
            correct = (pred == target) & valid
            tok_acc = correct.sum() / valid.sum()
            seq_acc = (correct | ~valid).all(1).float().mean()
        return loss, {"tok_acc": tok_acc.item(), "seq_acc": seq_acc.item()}

    @torch.no_grad()
    def generate(self, batch, **kw):
        mem, pad = self.encode(batch)
        return self.decoder.generate(mem, pad, **kw)

    def checkpoint(self) -> dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "Solid2Param":
        m = cls(ModelConfig(**ckpt["config"]))
        m.load_state_dict(ckpt["state_dict"])
        return m
