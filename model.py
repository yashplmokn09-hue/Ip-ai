"""
IP-AIv1 — Maximum Architecture
Implements: RoPE positional encoding, Grouped Query Attention (GQA),
SwiGLU activation, RMSNorm, weight tying.
~50M parameters at default config.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMSNorm ────────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ── Rotary Positional Encoding (RoPE) ─────────────────────────────────────
def precompute_rope(head_dim: int, max_seq: int, base: float = 10000.0):
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos   = torch.arange(max_seq).float()
    freqs = torch.outer(pos, theta)
    cos   = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin   = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin  # (max_seq, head_dim)


def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: (batch, heads, seq, head_dim)
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


# ── Grouped Query Attention (GQA) ─────────────────────────────────────────
class GQAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep      = n_heads // n_kv_heads
        self.head_dim   = d_model // n_heads

        self.q_proj  = nn.Linear(d_model, n_heads    * self.head_dim, bias=False)
        self.k_proj  = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj  = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj  = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Expand KV heads to match Q heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scale = math.sqrt(self.head_dim)
        attn  = (q @ k.transpose(-2, -1)) / scale
        mask  = torch.triu(torch.ones(t, t, device=x.device), diagonal=1).bool()
        attn  = attn.masked_fill(mask, float("-inf"))
        attn  = self.dropout(F.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o_proj(out)


# ── SwiGLU Feed-Forward ───────────────────────────────────────────────────
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * 8 / 3)
        hidden = (hidden + 63) // 64 * 64  # round up to multiple of 64
        self.gate  = nn.Linear(d_model, hidden, bias=False)
        self.up    = nn.Linear(d_model, hidden, bias=False)
        self.down  = nn.Linear(hidden, d_model, bias=False)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


# ── Transformer Block ─────────────────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = GQAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ff    = SwiGLU(d_model, dropout)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ff(self.norm2(x))
        return x


# ── IP-AIv1 Model ─────────────────────────────────────────────────────────
class IPAIv1(nn.Module):
    """
    IP-AIv1 Maximum Architecture
    Default: ~50M parameters
    """

    def __init__(
        self,
        vocab_size  : int   = 32000,
        block_size  : int   = 1024,
        d_model     : int   = 768,
        n_layers    : int   = 12,
        n_heads     : int   = 12,
        n_kv_heads  : int   = 4,     # GQA: 4 KV heads for 12 query heads
        dropout     : float = 0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.token_emb  = nn.Embedding(vocab_size, d_model)
        self.drop       = nn.Dropout(dropout)
        self.blocks     = nn.ModuleList([
            Block(d_model, n_heads, n_kv_heads, dropout) for _ in range(n_layers)
        ])
        self.norm_f     = RMSNorm(d_model)
        self.lm_head    = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        head_dim = d_model // n_heads
        cos, sin = precompute_rope(head_dim, block_size)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.block_size
        x   = self.drop(self.token_emb(idx))
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_f(x))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu"):
        ckpt  = torch.load(path, map_location=device)
        cfg   = ckpt["cfg"]
        model = cls(**{k: cfg[k] for k in
                       ["vocab_size","block_size","d_model","n_layers","n_heads","n_kv_heads","dropout"]})
        model.load_state_dict(ckpt["model"])
        return model.to(device)
