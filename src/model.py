import torch
import torch.nn as nn
import torch.nn.functional as F
from .tokenizer import VOCAB_SIZE
# MiniGPT causal: arquitetura anterior, baseada na estrutura do Hendrick.
class Head(nn.Module):
    def __init__(self, d: int, head_dim: int, max_T: int):
        super().__init__()
        self.key = nn.Linear(d, head_dim, bias=False)
        self.query = nn.Linear(d, head_dim, bias=False)
        self.value = nn.Linear(d, head_dim, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_T, max_T)))
        self.head_dim = head_dim

    def forward(self, x):
        _, T_cur, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:T_cur, :T_cur] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, d: int, h: int, max_T: int):
        super().__init__()
        assert d % h == 0
        head_dim = d // h
        self.heads = nn.ModuleList([Head(d, head_dim, max_T) for _ in range(h)])
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d: int, h: int, max_T: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadAttention(d, h, max_T)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = FeedForward(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class MazeMiniGPT(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d=64, max_T=128, L=2, h=4):
        super().__init__()
        self.max_T = max_T
        self.token_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(max_T, d)
        self.blocks = nn.ModuleList([TransformerBlock(d, h, max_T) for _ in range(L)])
        self.ln_final = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size)

    def forward(self, idx):
        _, T_cur = idx.shape
        pos = torch.arange(T_cur, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

