import torch.nn.functional as F

from torch import nn, Tensor, einsum
from einops import rearrange


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads

        self.qkv_embedding = nn.Linear(embedding_size, embedding_size * 3)
        self.projection = nn.Linear(embedding_size, embedding_size)

        self.attention_dropout = nn.Dropout(dropout)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        queries, keys, values = rearrange(self.qkv_embedding(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads,
                                          qkv=3)

        energy = einsum("bhqd, bhkd -> bhqk", queries, keys) / ((self.embedding_size / self.num_heads) ** (1 / 2))
        attn_score = F.softmax(energy, dim=-1)
        attn_score = self.attention_dropout(attn_score)

        out = einsum("bhqk, bhkv -> bhqv", attn_score, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        out = self.projection_dropout(out)

        return out
