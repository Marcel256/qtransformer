import torch.nn as nn
from transformer.causal_self_attention import CausalSelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, hidden_dim, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.attn = CausalSelfAttention(hidden_dim, n_heads)
        self.attn_dropout = nn.Dropout(dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, attn_mask):
        x = x + self.attn_dropout(self.attn(self.ln1(x), attn_mask)[0])
        return x + self.feed_forward(self.ln2(x))

    def forward_attn(self, x, attn_mask):
        attn_res, weights = self.attn(self.ln1(x), attn_mask)
        x = x + self.attn_dropout(attn_res)
        return x + self.feed_forward(self.ln2(x)), weights


class Transformer(nn.Module):

    def __init__(self, hidden_dim, n_heads, n_layer):
        super().__init__()
        self.layer = nn.ModuleList([TransformerBlock(hidden_dim, n_heads) for i in range(n_layer)])

    def forward(self, x, attn_mask=None):
        for layer in self.layer:
            x = layer(x, attn_mask)

        return x

    def forward_attn(self, x, attn_mask=None):
        weights = []
        for layer in self.layer:
            x,w = layer.forward_attn(x, attn_mask)
            weights.append(w)

        return x, weights
