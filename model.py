"""Decoder-only transformer (GPT-2 style, pre-norm).

Token and positional embeddings are summed, then passed through N blocks. Each
block applies masked multi-head self-attention and a feed-forward network, and
adds each result back onto the residual stream. A final LayerNorm and a linear
projection give logits over the vocabulary.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0
    fused_attention: bool = True  # use PyTorch's fused kernel; same math as the manual path


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.fused = cfg.fused_attention
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, self.head_dim).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v

        # The mask is only needed when several queries are processed at once; a
        # single new query during generation may look at every cached key.
        causal = T > 1
        if self.fused:
            y = F.scaled_dot_product_attention(
                q, k, v, is_causal=causal, dropout_p=self.dropout if self.training else 0.0)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if causal:
                S = k.size(2)
                mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(S - T)
                att = att.masked_fill(~mask, float("-inf"))
            att = self.attn_drop(F.softmax(att, dim=-1))
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.ff = FeedForward(cfg)

    def forward(self, x, cache=None):
        x = x + self.attn(self.ln1(x), cache)
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # Scale down the projections that write into the residual stream.
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None, start_pos=0):
        B, T = idx.shape
        assert start_pos + T <= self.cfg.block_size
        pos = torch.arange(start_pos, start_pos + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for i, block in enumerate(self.blocks):
            x = block(x, caches[i] if caches is not None else None)
        x = self.ln_f(x)
        if targets is None:
            return self.lm_head(x[:, -1:, :]), None
        logits = self.lm_head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, lr, weight_decay, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        groups = [
            {"params": [p for p in params if p.dim() >= 2], "weight_decay": weight_decay},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=device_type == "cuda")

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, top_p=0.95,
                 repetition_penalty=1.0, eos_id=None, generator=None):
        """Yield sampled token ids one at a time for a (1, T) prompt."""
        was_training = self.training
        self.eval()
        try:
            max_new_tokens = min(max_new_tokens, self.cfg.block_size - 1)
            idx = idx[:, -(self.cfg.block_size - max_new_tokens):]
            caches = [{} for _ in self.blocks]
            seen = idx[0].tolist()
            logits, _ = self(idx, caches=caches)
            pos = idx.size(1)
            for i in range(max_new_tokens):
                logits = logits[:, -1, :].float()
                if repetition_penalty != 1.0:
                    ids = torch.tensor(sorted(set(seen)), device=logits.device)
                    sel = logits[:, ids]
                    logits[:, ids] = torch.where(sel > 0, sel / repetition_penalty, sel * repetition_penalty)
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                if top_p is not None and top_p < 1.0:
                    sorted_p, order = probs.sort(dim=-1, descending=True)
                    drop = sorted_p.cumsum(dim=-1) - sorted_p >= top_p
                    sorted_p = sorted_p.masked_fill(drop, 0.0)
                    probs = torch.zeros_like(probs).scatter(-1, order, sorted_p)
                    probs = probs / probs.sum(dim=-1, keepdim=True)
                nxt = torch.multinomial(probs, 1, generator=generator)
                token = nxt.item()
                yield token
                if token == eos_id or i == max_new_tokens - 1:
                    break
                seen.append(token)
                logits, _ = self(nxt, caches=caches, start_pos=pos)
                pos += 1
        finally:
            self.train(was_training)
