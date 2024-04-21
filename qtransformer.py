import torch
import torch.nn as nn
from transformer.transformer import Transformer
import numpy as np

import matplotlib.pyplot as plt

class QTransformer(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, action_bins, seq_len, n_layers=3, n_heads=4, device=None):
        super().__init__()
        self.transformer = Transformer(hidden_dim, n_heads, n_layers)

        self.action_dim = action_dim

        self.state_emb = nn.Linear(state_dim, hidden_dim)

        self.advantage = nn.Linear(hidden_dim, action_bins)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.action_emb = nn.Embedding(action_bins, hidden_dim)

        mask_size = action_dim + seq_len - 1
        self.attn_mask = torch.from_numpy(np.logical_not(np.tril(np.ones((mask_size, mask_size)))))
        self.device = device
        if device:
            self.attn_mask = self.attn_mask.to(device)

        self.pos_enc = nn.Parameter(torch.randn((1, mask_size, hidden_dim))*0.1, requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, states, actions):
        state_token = self.state_emb(states)
        if self.action_dim > 1:
            a_token = self.action_emb(actions[:,:-1])
            token = torch.cat((state_token, a_token), dim=1)
        else:
            token = state_token
        token =  token + self.pos_enc
        out = self.transformer(token, attn_mask=self.attn_mask)
        action_token = out[:,-(self.action_dim):]
        adv = self.advantage(action_token)
        value = self.value_head(action_token)
        q_values = value + (adv - torch.mean(adv, dim=-1))

        return q_values


    def predict_action(self, states):
        actions = torch.zeros((states.shape[0], self.action_dim)).int().to(self.device)
        states = states.to(self.device)
        with torch.no_grad():
            for i in range(self.action_dim):
                q_values = self.forward(states, actions)
                indices = torch.argmax(q_values[:, -self.action_dim+i], dim=-1)
                actions[:, i] = indices

        return actions.cpu().numpy()