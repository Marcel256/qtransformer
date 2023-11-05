import torch
import torch.nn as nn
from transformer.transformer import Transformer
import numpy as np

class QTransformer(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, action_bins, seq_len, n_layers=3, n_heads=4):
        super().__init__()
        self.transformer = Transformer(hidden_dim, n_heads, n_layers)

        self.action_dim = action_dim

        self.state_emb = nn.Linear(state_dim, hidden_dim)

        self.q_head = nn.Linear(hidden_dim, action_bins, bias=False)
        self.action_emb = nn.Embedding(action_bins, hidden_dim)

        self.pos_enc = nn.Parameter(torch.randn((1, seq_len+action_dim, hidden_dim))*0.02, requires_grad=True)


    def forward(self, states, actions):

        state_token = self.state_emb(states)
        a_token = self.action_emb(actions)

        token = torch.cat((state_token, a_token), dim=1) + self.pos_enc
        out = self.transformer(token)
        q_values = self.q_head(out[:,-self.action_dim:])

        return torch.sigmoid(q_values)


    def predict_action(self, states):

        actions = torch.zeros((states.shape[0], self.action_dim)).int()
        with torch.no_grad():
            for i in range(self.action_dim):
                q_values = self.forward(states, actions)
                indices = torch.argmax(q_values[:,(-self.action_dim+i)], dim=-1)
                actions[:,i] = indices

        return actions.cpu().numpy()