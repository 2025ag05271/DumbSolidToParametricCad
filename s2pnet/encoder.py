"""B-Rep encoder: UV-Net style grid CNNs + edge-aware message passing + face Transformer.

Input (pure torch, no DGL): node_x (ΣF, N, N, 7), edge_x (ΣE, N, 6), edge_index (2, ΣE) directed,
node_batch (ΣF,) graph id per face. Output: padded face memory (B, Fmax+1, d) with a leading
global token, and a key-padding mask (True = padding).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class SurfaceCNN(nn.Module):
    """UV-Net surface encoder on the (7, N, N) face grid + explicit geometric summary."""

    def __init__(self, d: int):
        super().__init__()
        c = [7, 64, 128, 256]
        layers = []
        for a, b in zip(c, c[1:]):
            layers += [nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.LeakyReLU(0.2)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(256, d)
        self.geo = _mlp(13, d, d)

    @staticmethod
    def summary(x):  # x: (F, N, N, 7)
        pts, nrm, mask = x[..., :3], x[..., 3:6], x[..., 6:7]
        w = mask.sum((1, 2)).clamp_min(1.0)  # (F, 1)
        has = (mask.flatten(1).sum(1, keepdim=True) > 0).float()
        centroid = (pts * mask).sum((1, 2)) / w
        centroid = has * centroid + (1 - has) * pts.mean((1, 2))
        big = 1e4
        m = mask.bool() | (has[:, None, None, :] == 0)
        lo = torch.where(m, pts, torch.full_like(pts, big)).amin((1, 2))
        hi = torch.where(m, pts, torch.full_like(pts, -big)).amax((1, 2))
        mean_n = (nrm * mask).sum((1, 2)) / w
        frac = mask.mean((1, 2))
        return torch.cat([centroid, lo, hi, mean_n, frac], dim=1)  # (F, 13)

    def forward(self, x):
        h = self.conv(x.permute(0, 3, 1, 2))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h) + self.geo(self.summary(x))


class CurveCNN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        c = [6, 64, 128, 256]
        layers = []
        for a, b in zip(c, c[1:]):
            layers += [nn.Conv1d(a, b, 3, padding=1, bias=False), nn.BatchNorm1d(b), nn.LeakyReLU(0.2)]
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(256, d)
        self.geo = _mlp(12, d, d)

    def forward(self, x):  # (E, N, 6)
        h = self.conv(x.permute(0, 2, 1))
        h = F.adaptive_avg_pool1d(h, 1).flatten(1)
        n = x.shape[1]
        g = torch.cat([x[:, 0, :3], x[:, n // 2, :3], x[:, -1, :3], x[:, :, 3:].mean(1)], dim=1)
        return self.fc(h) + self.geo(g)


class EdgeConvLayer(nn.Module):
    """h_i ← h_i + MLP([h_i, Σ_j MLP([h_j, e_ij])]);  e_ij ← e_ij + MLP([h_i, h_j, e_ij])."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.msg = _mlp(2 * d, 2 * d, d)
        self.upd = _mlp(2 * d, 2 * d, d, dropout)
        self.eupd = _mlp(3 * d, 2 * d, d, dropout)
        self.nn_ = nn.LayerNorm(d)
        self.ne = nn.LayerNorm(d)

    def forward(self, h, e, edge_index):
        src, dst = edge_index
        hn, en = self.nn_(h), self.ne(e)
        m = self.msg(torch.cat([hn[src], en], dim=1))
        agg = torch.zeros_like(h).index_add_(0, dst, m.to(h.dtype))
        deg = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype).index_add_(
            0, dst, torch.ones(dst.shape[0], 1, device=h.device, dtype=h.dtype))
        agg = agg / deg.clamp_min(1.0)
        h = h + self.upd(torch.cat([hn, agg], dim=1))
        e = e + self.eupd(torch.cat([hn[dst], hn[src], en], dim=1))
        return h, e


class BRepEncoder(nn.Module):
    def __init__(self, d: int = 384, graph_layers: int = 4, tf_layers: int = 3, heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.surf = SurfaceCNN(d)
        self.curve = CurveCNN(d)
        self.gnn = nn.ModuleList(EdgeConvLayer(d, dropout) for _ in range(graph_layers))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, tf_layers, enable_nested_tensor=False)
        self.global_tok = nn.Parameter(torch.zeros(1, 1, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, node_x, edge_x, edge_index, node_batch, batch_size: int):
        h = self.surf(node_x)
        e = self.curve(edge_x)
        for layer in self.gnn:
            h, e = layer(h, e, edge_index)

        counts = torch.bincount(node_batch, minlength=batch_size)
        fmax = int(counts.max())
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.arange(h.shape[0], device=h.device) - starts[node_batch]
        mem = h.new_zeros(batch_size, fmax, h.shape[1])
        mem[node_batch, slot] = h
        pad = torch.ones(batch_size, fmax, dtype=torch.bool, device=h.device)
        pad[node_batch, slot] = False

        mem = torch.cat([self.global_tok.expand(batch_size, -1, -1).to(mem.dtype), mem], dim=1)
        pad = torch.cat([torch.zeros(batch_size, 1, dtype=torch.bool, device=h.device), pad], dim=1)
        mem = self.tf(mem, src_key_padding_mask=pad)
        return self.norm(mem), pad
