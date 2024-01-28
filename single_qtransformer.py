import torch
import torch.nn as nn
from transformer.transformer import Transformer
import numpy as np

import matplotlib.pyplot as plt

class QTransformer(nn.Module):

    def __init__(self, state_dim, actions, hidden_dim, seq_len, n_layers=3, n_heads=4):
        super().__init__()
        self.transformer = Transformer(hidden_dim, n_heads, n_layers)

        self.action_dim = actions

        self.state_emb = nn.Linear(state_dim, hidden_dim)

        self.q_head = nn.Linear(hidden_dim, actions, bias=False)

        mask_size = seq_len
        self.attn_mask = torch.from_numpy(np.logical_not(np.tril(np.ones((mask_size, mask_size)))))

        self.pos_enc = nn.Parameter(torch.randn((1, seq_len, hidden_dim))*0.02, requires_grad=True)


    def forward(self, states):

        state_token = self.state_emb(states)

        token = state_token + self.pos_enc
        out = self.transformer(token, attn_mask=self.attn_mask)
        q_values = self.q_head(out[:,-1])

        return q_values


    def predict_action(self, states):
        with torch.no_grad():
            q_values = self.forward(states)

        return torch.argmax(q_values, dim=-1).cpu().numpy()