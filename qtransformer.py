import torch
import torch.nn as nn
from transformer.transformer import Transformer
import numpy as np
from transformer.llama import LlamaModel

import matplotlib.pyplot as plt

class DuelingHead(nn.Module):

    def __init__(self, input_dim, output_dim, expansion_factor=2):
        super().__init__()
        self.hidden_dim = input_dim * expansion_factor
        self.stem = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU()
        )
        self.advantage = nn.Linear(self.hidden_dim, output_dim)
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        hidden = self.stem(x)
        adv = self.advantage(hidden)
        value = self.value_head(hidden)
        return value + (adv - torch.mean(adv, dim=-1, keepdim=True))


class QTransformer(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim, action_bins, seq_len, n_layers=3, n_heads=4, max_ep_len=1000, dueling=False, device=None):
        super().__init__()

        mask_size = action_dim + seq_len - 1
        self.transformer = LlamaModel(hidden_dim, num_heads=n_heads, max_position=mask_size, num_layers=n_layers)#Transformer(hidden_dim, n_heads, n_layers)

        self.action_dim = action_dim

        self.state_emb = nn.Linear(state_dim, hidden_dim)

        if dueling:
            self.out = DuelingHead(hidden_dim, action_bins)
        else:
            self.out = nn.Linear(hidden_dim, action_bins)

        self.action_emb = nn.Embedding(action_bins, hidden_dim)
        self.time_emb = nn.Embedding(max_ep_len, hidden_dim)


        tri_mask = np.tril(np.ones((mask_size, mask_size)))
        tri_mask[:seq_len, :seq_len] = 1
        self.attn_mask = torch.from_numpy(np.logical_not(tri_mask))
        self.attn_mask = self.attn_mask * torch.finfo(torch.float32).min
        self.attn_mask = self.attn_mask.unsqueeze(0).unsqueeze(0)
        self.device = device
        self.pos_ids = torch.arange(0, mask_size).long().unsqueeze(0)
        if device:
            self.attn_mask = self.attn_mask.to(device)
            self.pos_ids = self.pos_ids.to(device)

        #self.pos_enc = nn.Parameter(torch.randn((1, mask_size, hidden_dim))*0.02, requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, states, actions, timesteps):
        time = self.time_emb(timesteps)
        state_token = self.state_emb(states) + time
        if self.action_dim > 1:
            a_token = self.action_emb(actions[:,:-1])
            token = torch.cat((state_token, a_token), dim=1)
        else:
            token = state_token
        #token =  token + self.pos_enc
        out = self.transformer(token, attention_mask=self.attn_mask, position_ids=self.pos_ids)
        action_token = out[:,-(self.action_dim):]
        q_values = self.out(action_token)

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