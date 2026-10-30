"""Autoregressive Transformer decoder with KV cache and grammar-constrained generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadseq.grammar import Grammar
from cadseq.tokenizer import BOS_ID, EOS_ID, PAD_ID, VOCAB


class _Attn(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)
        self.dropout = dropout

    def split(self, x):  # (B, L, d) → (B, h, L, dk)
        return x.view(x.shape[0], x.shape[1], self.h, self.dk).transpose(1, 2)

    def kv_of(self, x):
        k, v = self.kv(x).chunk(2, dim=-1)
        return self.split(k), self.split(v)

    def forward(self, x, k, v, mask=None, causal=False):
        q = self.split(self.q(x))
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal,
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o(y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1))


class DecoderLayer(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.self_attn = _Attn(d, heads, dropout)
        self.cross_attn = _Attn(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem_kv, mem_mask, cache=None):
        h = self.n1(x)
        k, v = self.self_attn.kv_of(h)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
            x = x + self.drop(self.self_attn(h, k, v))  # one new position attends to all cached
        else:
            x = x + self.drop(self.self_attn(h, k, v, causal=True))
        x = x + self.drop(self.cross_attn(self.n2(x), *mem_kv, mask=mem_mask))
        return x + self.drop(self.ff(self.n3(x)))


class SeqDecoder(nn.Module):
    def __init__(self, d=384, layers=8, heads=8, dropout=0.1, max_len=1024, vocab=len(VOCAB)):
        super().__init__()
        self.tok = nn.Embedding(vocab, d, padding_idx=PAD_ID)
        self.pos = nn.Embedding(max_len, d)
        self.layers = nn.ModuleList(DecoderLayer(d, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight  # tied embeddings
        self.max_len = max_len
        self.d = d
        # GPT-style init: small embeddings keep the tied output logits well-scaled at start.
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)
        with torch.no_grad():
            self.tok.weight[PAD_ID].zero_()

    def embed(self, tokens, positions):
        return self.tok(tokens) + self.pos(positions)

    def memory_kv(self, mem):
        return [layer.cross_attn.kv_of(mem) for layer in self.layers]

    @staticmethod
    def memory_mask(mem_pad):  # (B, M) True = pad → additive-free boolean mask (B,1,1,M), True = keep
        return (~mem_pad)[:, None, None, :]

    def forward(self, tokens, mem, mem_pad):
        """Teacher forcing: tokens (B, L) → logits (B, L, V) predicting tokens shifted by one."""
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens, pos[None])
        kvs, mmask = self.memory_kv(mem), self.memory_mask(mem_pad)
        for layer, kv in zip(self.layers, kvs):
            x = layer(x, kv, mmask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, mem, mem_pad, max_len=None, constrained=True, temperature=0.0,
                 canonical_order=True, generator=None):
        """Greedy (temperature 0) or sampled decoding. Returns a list of token-id lists."""
        b, dev = mem.shape[0], mem.device
        max_len = min(max_len or self.max_len, self.max_len)
        kvs, mmask = self.memory_kv(mem), self.memory_mask(mem_pad)
        caches = [{} for _ in self.layers]
        grammars = [Grammar(canonical_order) for _ in range(b)] if constrained else None
        out = [[BOS_ID] for _ in range(b)]
        if grammars:
            for g in grammars:
                g.feed(BOS_ID)
        alive = torch.ones(b, dtype=torch.bool, device=dev)
        cur = torch.full((b, 1), BOS_ID, dtype=torch.long, device=dev)
        mask_cache: dict = {}
        vocab = self.head.out_features

        for t in range(max_len - 1):
            x = self.embed(cur, torch.full_like(cur, t))
            for layer, kv, cache in zip(self.layers, kvs, caches):
                x = layer(x, kv, mmask, cache)
            logits = self.head(self.norm(x[:, -1])).float()
            if grammars:
                allow = torch.zeros(b, vocab, dtype=torch.bool, device=dev)
                for i, g in enumerate(grammars):
                    if g.done:
                        allow[i, PAD_ID] = True
                        continue
                    key = g.allowed
                    m = mask_cache.get(key)
                    if m is None:
                        m = torch.zeros(vocab, dtype=torch.bool, device=dev)
                        m[list(key)] = True
                        mask_cache[key] = m
                    allow[i] = m
                logits = logits.masked_fill(~allow, float("-inf"))
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = torch.where(alive, nxt, torch.full_like(nxt, PAD_ID))
            ids = nxt.tolist()
            for i, tok in enumerate(ids):
                if not alive[i]:
                    continue
                out[i].append(tok)
                if grammars:
                    grammars[i].feed(tok)
                if tok == EOS_ID:
                    alive[i] = False
            if not alive.any():
                break
            cur = nxt[:, None]
        return out
