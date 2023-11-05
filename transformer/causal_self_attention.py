import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(self, hidden_dim, n_heads):
        super().__init__()
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)

    def forward(self, x, attn_mask):
        batch_size, seq_len, hidden_dim = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        return self.attn(q, k, v, attn_mask=attn_mask)