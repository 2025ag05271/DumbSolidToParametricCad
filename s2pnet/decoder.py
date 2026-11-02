"""Autoregressive Transformer decoder with KV cache and grammar-constrained generation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cadseq.grammar import Grammar
from cadseq.tokenizer import BOS_ID, COARSE_BASE, EOS_ID, NUM_COARSE, NUM_OFFSET, PAD_ID, VOCAB


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

    def forward(self, tokens, mem, mem_pad, return_hidden: bool = False):
        """Teacher forcing: tokens (B, L) → logits (B, L, V) predicting tokens shifted by one
        (and the final hidden states (B, L, d) if ``return_hidden``)."""
        pos = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens, pos[None])
        kvs, mmask = self.memory_kv(mem), self.memory_mask(mem_pad)
        for layer, kv in zip(self.layers, kvs):
            x = layer(x, kv, mmask)
        h = self.norm(x)
        return (self.head(h), h) if return_hidden else self.head(h)

    @torch.no_grad()
    def generate(self, mem, mem_pad, max_len=None, constrained=True, temperature=0.0,
                 canonical_order=True, generator=None, copy=None):
        """Greedy (temperature 0) or sampled decoding. Returns a list of token-id lists.

        ``copy`` (optional, see ``Solid2Param.copy_context``): when a number is about to start, the
        copy gate decides between copying an input candidate value (emitted as its two tokens) and
        generating the digits."""
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
        forced = torch.full((b,), -1, dtype=torch.long, device=dev)  # fine token of a copied number
        self.copy_stats = {"numbers": 0, "copied": 0}

        for t in range(max_len - 1):
            x = self.embed(cur, torch.full_like(cur, t))
            for layer, kv, cache in zip(self.layers, kvs, caches):
                x = layer(x, kv, mmask, cache)
            h = self.norm(x[:, -1])
            logits = self.head(h).float()
            raw_num = torch.logsumexp(logits[:, COARSE_BASE:COARSE_BASE + NUM_COARSE], -1)  # P(number) as trained
            allow = None
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
            if copy is not None:
                # "number vs other token" is decided on the total number mass over *all* digit buckets
                # (the copy model's training definition); the grammar's value range is enforced by the
                # copy head / the digit choice afterwards, not by masking that mass away.
                cl = logits[:, COARSE_BASE:COARSE_BASE + NUM_COARSE]
                num_ok = torch.isfinite(cl).any(-1)
                num_mass = torch.where(num_ok, raw_num, torch.full_like(raw_num, float("-inf")))
                other = logits.clone()
                other[:, COARSE_BASE:COARSE_BASE + NUM_COARSE] = float("-inf")
                if temperature > 0:
                    joint = torch.cat([other, num_mass[:, None]], -1) / temperature
                    pick = torch.multinomial(torch.softmax(joint, -1), 1, generator=generator).squeeze(1)
                    digit = torch.multinomial(torch.softmax(cl.masked_fill(~num_ok[:, None], 0.0) / temperature, -1),
                                              1, generator=generator).squeeze(1)
                    nxt = torch.where(pick == other.shape[1], COARSE_BASE + digit, pick)
                else:
                    best_other, other_idx = other.max(-1)
                    nxt = torch.where(num_mass > best_other, COARSE_BASE + cl.argmax(-1), other_idx)
            elif temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            if copy is not None:
                starts = alive & (forced < 0) & (nxt >= COARSE_BASE) & (nxt < COARSE_BASE + NUM_COARSE)
                rows = torch.nonzero(starts).squeeze(1)
                if len(rows):
                    r_list = rows.tolist()
                    ranges = [(grammars[i].number_range if grammars else None) or (-NUM_OFFSET, NUM_OFFSET - 1)
                              for i in r_list]
                    lo = torch.tensor([r[0] for r in ranges], device=dev)
                    hi = torch.tensor([r[1] for r in ranges], device=dev)
                    coarse, fine = copy.choose(h, rows, lo, hi, temperature, generator)
                    self.copy_stats["numbers"] += len(rows)
                    hit = coarse >= 0
                    self.copy_stats["copied"] += int(hit.sum())
                    nxt[rows[hit]] = coarse[hit]
                    forced[rows[hit]] = fine[hit]
            use_forced = forced >= 0
            if copy is not None:
                is_fine_step = use_forced & ~((nxt >= COARSE_BASE) & (nxt < COARSE_BASE + NUM_COARSE))
                nxt = torch.where(is_fine_step, forced, nxt)
                forced = torch.where(is_fine_step, torch.full_like(forced, -1), forced)
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
