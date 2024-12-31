import copy

from torch import nn
import torch

from qtransformer.loss.loss import Loss
from qtransformer.model.qtransformer import QTransformer
from qtransformer.trainer_config import TrainConfig

def norm_rewards(r, R_min, R_max):
    return (r - R_min) / (R_max - R_min)

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

class QLoss(Loss):

    def __init__(self, model: QTransformer, config: TrainConfig):
        super.__init__()
        self.model = model
        self.target_model = copy.deepcopy(self.model)
        self.config = config
        self.gamma = config.gamma
        self.mc_return = config.mc_returns
        self.n_step = False
        self.reg_weight = config.reg_weight

        self.loss = nn.MSELoss()
        self.R_min = config.R_min
        self.R_max = config.R_max


    def forward(self, states, actions, rewards, returns, terminals, timesteps):
        soft_update(self.model, self.target_model, self.config.tau)
        with torch.no_grad():
            q_next = torch.max(torch.sigmoid(self.target_model(states[:, 1:], actions[:, -1], timesteps[:, 1:])[:, 0].unsqueeze(1)),
                      dim=2, keepdim=True)[0]

        q = torch.sigmoid(self.model(states[:, :-1], actions[:, -2], timesteps[:, :-1]))

        r = rewards.unsqueeze(2)[:, -2].unsqueeze(1) / (self.R_max - self.R_min)

        mc_returns_next = norm_rewards(returns.unsqueeze(2)[:, -1].unsqueeze(1), self.R_min, self.R_max)
        q_target_next = r + self.gamma * (1 - terminals[:, -2].unsqueeze(1)) * q_next
        if self.mc_returns:
            next_timestep = torch.maximum(q_target_next, mc_returns_next)
        else:
            next_timestep = q_target_next

        if self.model.action_dim > 1:
            mc_returns_curr = norm_rewards(returns.unsqueeze(2)[:, -2].unsqueeze(1), self.R_min, self.R_max)
            q_target_curr = torch.max(q[:, 1:], dim=2, keepdim=True)[0]
            if self.mc_returns:
                curr_timestep = torch.maximum(q_target_curr, mc_returns_curr)
            else:
                curr_timestep = q_target_curr
            next_dim = torch.cat([curr_timestep, next_timestep], dim=1)
        else:
            next_dim = next_timestep

        pred = torch.gather(q, 2, actions[:, -2].unsqueeze(2).long())

        action_mask = torch.ones_like(q)
        action_mask.scatter_(2, actions[:, -2].unsqueeze(2).long(), 0)

        bin_sum = torch.sum((q ** 2) * action_mask)

        reg_loss = bin_sum / action_mask.sum()

        td_loss = self.loss(pred, next_dim.detach())

        return (td_loss + self.reg_weight * reg_loss) / 2
