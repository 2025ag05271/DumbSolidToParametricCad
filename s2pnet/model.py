"""Full encoder–decoder model (v1: tokens only; v2: + exact attributes, copy head, aux segmentation)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadseq.tokenizer import COARSE_BASE, FINE_BASE, NUM_COARSE, NUM_FINE, NUM_OFFSET, PAD_ID

from .decoder import SeqDecoder
from .encoder import BRepEncoder, _mlp


@dataclass
class ModelConfig:
    d_model: int = 384
    heads: int = 8
    graph_layers: int = 4
    enc_layers: int = 3
    dec_layers: int = 8
    dropout: float = 0.1
    max_len: int = 1024
    # v2 (defaults keep v1 checkpoints loadable)
    attrs: bool = False      # exact analytic face / edge attributes as encoder inputs
    copy: bool = False       # pointer-generator: copy numbers from input candidate values
    aux_seg: float = 0.0     # weight of the auxiliary per-face feature-class loss
    edge_ptr: bool = False   # phase 3b: fillet / chamfer edge refs point at virtual (sharp) edges


class CopyHead(nn.Module):
    """Embeds the input's candidate values (value Fourier features + provenance tags, contextualised by
    cross-attention to the B-Rep face memory) and scores them against decoder states."""

    def __init__(self, d: int, feat_dim: int, heads: int, dropout: float):
        super().__init__()
        self.inp = _mlp(feat_dim, d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.cross = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = _mlp(d, 2 * d, d, dropout)
        self.k = nn.Linear(d, d)
        self.q = nn.Linear(d, d)
        self.gate = nn.Linear(d, 1)
        self.scale = 1.0 / math.sqrt(d)

    def keys(self, cand_feat, mem, mem_pad):
        c = self.inp(cand_feat)
        c = c + self.cross(self.n1(c), mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        c = c + self.ff(self.n2(c))
        return self.k(c)  # (B, C, d)

    def scores(self, h, keys, cand_pad):
        """h (B, L, d), keys (B, C, d) → (B, L, C) logits, pads −inf."""
        s = torch.einsum("bld,bcd->blc", self.q(h), keys).float() * self.scale
        return s.masked_fill(cand_pad[:, None, :], float("-inf"))


class EdgeHead(CopyHead):
    """Pointer over the input's virtual edges (``cadseq.blends``): keys from the sharp-edge point /
    curve type plus the embeddings of the blend face and its two support faces."""

    def __init__(self, d: int, feat_dim: int, heads: int, dropout: float):
        super().__init__(d, feat_dim, heads, dropout)
        self.faces = nn.Linear(3 * d, d)

    def keys(self, feat, face, sup, mem, mem_pad):
        idx = torch.cat([face[..., None], sup], -1) + 1  # memory slot 0 is the [GLOBAL] token
        idx = idx.clamp(max=mem.shape[1] - 1)
        fe = torch.gather(mem, 1, idx.reshape(idx.shape[0], -1, 1).expand(-1, -1, mem.shape[-1]))
        c = self.inp(feat) + self.faces(fe.reshape(*idx.shape[:2], -1))
        c = c + self.cross(self.n1(c), mem, mem, key_padding_mask=mem_pad, need_weights=False)[0]
        c = c + self.ff(self.n2(c))
        return self.k(c)


class _CopyContext:
    """Per-batch copy state for ``SeqDecoder.generate``."""

    def __init__(self, head: CopyHead, keys, cand_pad, cand_quant, edge=None):
        self.head, self.keys, self.quant = head, keys, cand_quant
        n = cand_quant + NUM_OFFSET
        self.valid_range = (n >= 0) & (n < 2 * NUM_OFFSET) & ~cand_pad
        n = n.clamp(0, 2 * NUM_OFFSET - 1)
        self.coarse = COARSE_BASE + n // NUM_FINE
        self.fine = FINE_BASE + n % NUM_FINE
        self.edge = edge  # (EdgeHead, keys, ve_pad, ve_quant, ve_curve) or None
        if edge is not None:
            _, _, ve_pad, ve_quant, _ = edge
            n = ve_quant + NUM_OFFSET  # (B, V, 3)
            self.ve_valid = ((n >= 0) & (n < 2 * NUM_OFFSET)).all(-1) & ~ve_pad
            n = n.clamp(0, 2 * NUM_OFFSET - 1)
            # (B, V, 6): coarse/fine token pairs of x, y, z
            self.ve_tokens = torch.stack([COARSE_BASE + n // NUM_FINE, FINE_BASE + n % NUM_FINE], -1).flatten(-2)

    def choose_edge(self, h, rows, curve, used, temperature, generator):
        """For batch ``rows`` that just emitted an edge-type token (curve index ``curve``): index of the
        virtual edge to point at, or −1 to fall back to copying / generating the coordinates.
        ``used`` (B, V) masks edges already referenced."""
        head, keys, _, _, ve_curve = self.edge
        hr = h[rows]
        gate = torch.sigmoid(head.gate(hr).squeeze(-1).float())
        q = head.q(h)
        s = (torch.einsum("bd,bcd->bc", q, keys).float() * head.scale)[rows]
        valid = self.ve_valid[rows] & (ve_curve[rows] == curve[:, None]) & ~used[rows]
        s = s.masked_fill(~valid, float("-inf"))
        has = torch.isfinite(s.max(-1).values)
        if temperature > 0:
            take = torch.rand(len(rows), device=h.device, generator=generator) < gate
            p = torch.softmax(s / temperature, dim=-1).nan_to_num(0.0)
            idx = torch.multinomial(torch.where(has[:, None], p, torch.ones_like(p)), 1, generator=generator).squeeze(1)
        else:
            take = gate > 0.5
            idx = s.argmax(-1)
        return torch.where(take & has, idx, torch.full_like(idx, -1))

    def choose(self, h, rows, lo, hi, temperature, generator):
        """For batch ``rows`` about to start a number with value bounds [lo, hi] (quantized, from the
        grammar): (coarse, fine) tokens of the copied candidate, or −1 to generate the digits."""
        hr = h[rows]
        gate = torch.sigmoid(self.head.gate(hr).squeeze(-1).float())
        q = self.head.q(h)  # all rows: avoids gathering the (B, C, d) keys
        s = (torch.einsum("bd,bcd->bc", q, self.keys).float() * self.head.scale)[rows]
        coarse, fine, quant = self.coarse[rows], self.fine[rows], self.quant[rows]
        valid = self.valid_range[rows] & (quant >= lo[:, None]) & (quant <= hi[:, None])
        s = s.masked_fill(~valid, float("-inf"))
        if temperature > 0:
            take = torch.rand(len(rows), device=h.device, generator=generator) < gate
            p = torch.softmax(s / temperature, dim=-1).nan_to_num(0.0)
            has = p.sum(-1) > 0
            idx = torch.multinomial(torch.where(has[:, None], p, torch.ones_like(p)), 1, generator=generator).squeeze(1)
        else:
            take = gate > 0.5
            idx = s.argmax(-1)
            has = torch.isfinite(s.max(-1).values)
        take = take & has
        c = torch.gather(coarse, 1, idx[:, None]).squeeze(1)
        f = torch.gather(fine, 1, idx[:, None]).squeeze(1)
        return torch.where(take, c, torch.full_like(c, -1)), torch.where(take, f, torch.full_like(f, -1))


class Solid2Param(nn.Module):
    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg
        fa = ea = 0
        if cfg.attrs:
            from step_to_graph import EDGE_ATTR_DIM, FACE_ATTR_DIM

            fa, ea = FACE_ATTR_DIM, EDGE_ATTR_DIM
        self.encoder = BRepEncoder(cfg.d_model, cfg.graph_layers, cfg.enc_layers, cfg.heads, cfg.dropout, fa, ea)
        self.decoder = SeqDecoder(cfg.d_model, cfg.dec_layers, cfg.heads, cfg.dropout, cfg.max_len)
        if cfg.copy:
            from cadseq.candidates import N_TAGS, VALUE_FEAT_DIM

            self.copy_head = CopyHead(cfg.d_model, VALUE_FEAT_DIM + N_TAGS, cfg.heads, cfg.dropout)
        if cfg.edge_ptr:
            from cadseq.candidates import VALUE_FEAT_DIM
            from cadseq.edgefix import CURVES

            self.edge_head = EdgeHead(cfg.d_model, 3 * VALUE_FEAT_DIM + len(CURVES), cfg.heads, cfg.dropout)
        if cfg.aux_seg > 0:
            from cadseq.provenance import FACE_CLASSES

            self.seg_head = _mlp(cfg.d_model, cfg.d_model, len(FACE_CLASSES), cfg.dropout)

    def encode(self, batch, return_nodes: bool = False):
        return self.encoder(batch["node_x"], batch["edge_x"], batch["edge_index"],
                            batch["node_batch"], batch["tokens"].shape[0],
                            batch.get("face_attr") if self.cfg.attrs else None,
                            batch.get("edge_attr") if self.cfg.attrs else None, return_nodes=return_nodes)

    def forward(self, batch):
        mem, pad = self.encode(batch)
        return self.decoder(batch["tokens"][:, :-1], mem, pad)

    def loss(self, batch, label_smoothing: float = 0.0):
        if not (self.cfg.copy or self.cfg.aux_seg > 0):
            return self._loss_v1(batch, label_smoothing)
        mem, pad, nodes = self.encode(batch, return_nodes=True)
        tokens = batch["tokens"]
        logits, hidden = self.decoder(tokens[:, :-1], mem, pad, return_hidden=True)
        target = tokens[:, 1:]
        logp = F.log_softmax(logits.float(), -1)
        nll = -logp.gather(-1, target[..., None]).squeeze(-1)  # (B, L-1)
        valid = target != PAD_ID
        metrics = {}
        with torch.no_grad():
            correct = (logits.argmax(-1) == target) & valid
            metrics["tok_acc"] = (correct.sum() / valid.sum()).item()
            metrics["seq_acc"] = (correct | ~valid).all(1).float().mean().item()

        if self.cfg.copy:
            nb, npos = batch["num_b"], batch["num_pos"]  # number = tokens[nb, npos] (coarse), +1 (fine)
            is_num = torch.zeros_like(valid)
            is_num[nb, npos - 1] = True  # predicting the coarse token
            is_num[nb, npos] = True      # predicting the fine token
            lp_gen = logp[nb, npos - 1, tokens[nb, npos]] + logp[nb, npos, tokens[nb, npos + 1]]
            # P(a number starts here) = total mass of the coarse tokens (vs e.g. THROUGH)
            lp_start = torch.logsumexp(logp[nb, npos - 1, COARSE_BASE:COARSE_BASE + NUM_COARSE], -1)
            h = hidden[nb, npos - 1]
            g = self.copy_head.gate(h).squeeze(-1).float()
            keys = self.copy_head.keys(batch["cand_feat"], mem, pad)
            # scores only at number starts, one part at a time (numbers are grouped by part):
            # avoids a (B, L, C) tensor
            q = self.copy_head.q(h)
            s = torch.full((len(nb), keys.shape[1]), float("-inf"), device=h.device)
            counts = torch.bincount(nb, minlength=keys.shape[0]).tolist()
            start = 0
            for b, n in enumerate(counts):
                if n:
                    s[start:start + n] = (q[start:start + n] @ keys[b].T).float() * self.copy_head.scale
                start += n
            s = s.masked_fill(batch["cand_pad"][nb], float("-inf"))
            logpi = F.log_softmax(s, -1)
            match = torch.zeros_like(s, dtype=torch.bool)
            match[batch["pair_num"], batch["pair_cand"]] = True
            lse = torch.logsumexp(logpi.masked_fill(~match, float("-inf")), -1)
            lp_num = torch.logaddexp(lp_start + F.logsigmoid(g) + lse, F.logsigmoid(-g) + lp_gen)
            other = valid & ~is_num
            loss = (nll[other].sum() - lp_num.sum()) / (other.sum() + len(lp_num))
            with torch.no_grad():
                metrics["struct_acc"] = (correct[other].sum() / other.sum()).item()
                has = match.any(-1)
                pick_ok = match.gather(1, s.argmax(-1, keepdim=True)).squeeze(1)
                gen_ok = (logits[nb, npos - 1].argmax(-1) == tokens[nb, npos]) & \
                         (logits[nb, npos].argmax(-1) == tokens[nb, npos + 1])
                copy = g > 0
                metrics["num_acc"] = torch.where(copy, pick_ok, gen_ok).float().mean().item()
                metrics["copy_rate"] = copy.float().mean().item()
                metrics["copyable"] = has.float().mean().item()
                metrics["copy_pick_acc"] = (pick_ok[has].float().mean().item() if has.any() else 0.0)
            if self.cfg.edge_ptr and len(batch["er_pos"]):
                ptr, ptr_metrics = self._edge_loss(batch, hidden, mem, pad)
                loss = loss + ptr / (other.sum() + len(lp_num))
                metrics.update(ptr_metrics)
        else:
            loss = nll[valid].mean()

        if self.cfg.aux_seg > 0 and "face_cls" in batch:
            seg = F.cross_entropy(self.seg_head(nodes).float(), batch["face_cls"])
            loss = loss + self.cfg.aux_seg * seg
            metrics["seg_loss"] = seg.item()
        return loss, metrics

    def _loss_v1(self, batch, label_smoothing):
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

    def _edge_loss(self, batch, hidden, mem, pad):
        """Summed pointer NLL over edge refs: gate·π(any virtual edge of the ref's group) when the ref
        has targets, else the gate must decline (fallback to number copying / generation)."""
        eb, epos = batch["er_b"], batch["er_pos"]
        head = self.edge_head
        h = hidden[eb, epos - 1]
        g = head.gate(h).squeeze(-1).float()
        keys = head.keys(batch["ve_feat"], batch["ve_face"], batch["ve_sup"], mem, pad)
        q = head.q(h)
        s = torch.full((len(eb), keys.shape[1]), float("-inf"), device=h.device)
        start = 0
        for b, n in enumerate(torch.bincount(eb, minlength=keys.shape[0]).tolist()):
            if n:
                s[start:start + n] = (q[start:start + n] @ keys[b].T).float() * head.scale
            start += n
        valid = ~batch["ve_pad"][eb] & (batch["ve_curve"][eb] == batch["er_curve"][:, None])
        s = s.masked_fill(~valid, float("-inf"))
        match = torch.zeros_like(s, dtype=torch.bool)
        match[batch["er_pair_ref"], batch["er_pair_ve"]] = True
        match &= valid
        has = match.any(-1)
        logpi = F.log_softmax(s.masked_fill(~valid.any(-1, keepdim=True), 0.0), -1)
        lp = torch.logsumexp(logpi.masked_fill(~match, float("-inf")), -1)
        nll = torch.where(has, -(F.logsigmoid(g) + lp.nan_to_num(0.0, neginf=0.0)), -F.logsigmoid(-g))
        with torch.no_grad():
            pick = match.gather(1, s.argmax(-1, keepdim=True)).squeeze(1)
            metrics = {"ptr_has": has.float().mean().item(),
                       "ptr_rate": (g > 0).float().mean().item(),
                       "ptr_acc": ((pick & (g > 0))[has].float().mean().item() if has.any() else 0.0)}
        return nll.sum(), metrics

    def copy_context(self, batch, mem, pad):
        if not self.cfg.copy:
            return None
        keys = self.copy_head.keys(batch["cand_feat"], mem, pad)
        edge = None
        if self.cfg.edge_ptr and "ve_feat" in batch:
            ekeys = self.edge_head.keys(batch["ve_feat"], batch["ve_face"], batch["ve_sup"], mem, pad)
            edge = (self.edge_head, ekeys, batch["ve_pad"], batch["ve_quant"], batch["ve_curve"])
        return _CopyContext(self.copy_head, keys, batch["cand_pad"], batch["cand_quant"], edge)

    @torch.no_grad()
    def generate(self, batch, **kw):
        mem, pad = self.encode(batch)
        return self.decoder.generate(mem, pad, copy=self.copy_context(batch, mem, pad), **kw)

    def checkpoint(self) -> dict:
        return {"config": asdict(self.cfg), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, ckpt: dict) -> "Solid2Param":
        m = cls(ModelConfig(**ckpt["config"]))
        m.load_state_dict(ckpt["state_dict"])
        return m
