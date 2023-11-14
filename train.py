import torch
from torch.nn import MSELoss

from qtransformer import QTransformer
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader


import hydra
from omegaconf import DictConfig

import numpy as np

from eval import eval
import wandb

import matplotlib.pyplot as plt

def norm_rewards(r, R_min, R_max):
    return r/R_max


def norm_rewards_2(r, R_min, R_max):
    return (r-R_min)/(R_max-R_min)

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:
    wandb.init(entity="marcel98", project="qtransformer")
    dataset = SequenceDataset('data/random_seq.pkl', 4)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


    action_dim = 6
    gamma = 0.99
    R_max = 600
    R_min = -545
    model = QTransformer(17, 6, 256, 256, 4)
    target_model = QTransformer(17, 6, 256, 256, 4)
    target_model.eval()
    soft_update(model, target_model, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss = MSELoss()

    log_loss_steps = 50
    eval_steps = 500
    loss_list = []
    td_loss_list = []
    reg_loss_list = []
    i = 0
    epochs = 1
    wandb.log({"eval_score": eval('HalfCheetah-v4', model, 10)})
    for epoch in range(epochs):
        for batch in dataloader:
            states, actions, rewards, returns, terminal = batch

            returns = returns.float()
            terminal = terminal.unsqueeze(2).float()
            with torch.no_grad():
                q_next = torch.max(target_model(states[:, 1:].float(), actions[:,-1].long())[:, 0].unsqueeze(1), dim=2, keepdim=True)[0]

            q = model(states[:, :-1].float(), actions[:,-2].long())

            r = norm_rewards(rewards.float().unsqueeze(2)[:,-2].unsqueeze(1), R_min, R_max)

            mc_returns_curr = norm_rewards(returns.unsqueeze(2)[:,-2].unsqueeze(1), R_min, R_max)
            mc_returns_next = norm_rewards(returns.unsqueeze(2)[:,-1].unsqueeze(1), R_min, R_max)
            next_timestep = torch.maximum(r + gamma * (1-terminal[:,-2].unsqueeze(1))*q_next, mc_returns_next)
            curr_timestep = torch.maximum(torch.max(q[:, 1:], dim=2, keepdim=True)[0], mc_returns_curr)

            next_dim = torch.cat([curr_timestep, next_timestep], dim=1)

            pred = torch.gather(q, 2, actions[:,-2].unsqueeze(2).long())

            action_mask = torch.ones_like(q)
            action_mask.scatter_(2, actions[:,-2].unsqueeze(2).long(), 0)

            bin_sum = torch.sum((q - (-1)) * action_mask, dim=-1, keepdim=True)

            reg_loss = torch.mean(bin_sum / (q.shape[-1] - 1))

            td_loss = loss(pred, next_dim.detach())

            err = 0.5 * td_loss + 0.5 * reg_loss
            optimizer.zero_grad()
            err.backward()
            loss_list.append(err.item())
            td_loss_list.append(td_loss.item())
            reg_loss_list.append(reg_loss.item())
            optimizer.step()
            if (i+1) % log_loss_steps == 0:
                wandb.log({"train_loss": np.mean(loss_list),
                           "td_loss": np.mean(td_loss_list),
                           "reg_loss": np.mean(reg_loss_list)})
                loss_list = []
                td_loss_list = []
                reg_loss_list = []
                soft_update(model, target_model, 0.1)

            if (i+1) % eval_steps == 0:
                model.eval()
                wandb.log({"eval_score": eval('HalfCheetah-v4', model, 10)})
                torch.save({'model_state': model.state_dict()}, 'models/model-{}.pt'.format(i))
                model.train()
            i += 1





if __name__ == "__main__":
    train()