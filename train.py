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

def norm_rewards(r, R_min, R_max):
    return (r-R_min)/(R_max-R_min)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:
    wandb.init(entity="marcel98", project="qtransformer")
    dataset = SequenceDataset('data/random_seq.pkl', 4)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)


    action_dim = 6
    gamma = 0.99
    R_max = 0
    R_min = -545
    model = QTransformer(17, 6, 256, 256, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = MSELoss()

    log_loss_steps = 100
    eval_steps = 500
    loss_list = []
    i = 0
    epochs = 1
    for epoch in range(epochs):
        for batch in dataloader:
            states, actions, rewards, returns, terminal = batch

            returns = returns.float()
            with torch.no_grad():
                q_next = torch.max(model(states[:, 1:].float(), actions[:,-1].long())[:, 0].unsqueeze(1), dim=2, keepdim=True)[0]

            q = model(states[:, :-1].float(), actions[:,-2].long())

            r = norm_rewards(rewards.float().unsqueeze(2)[:,-2].unsqueeze(1), R_min, R_max)

            mc_returns_curr = norm_rewards(returns.unsqueeze(2)[:,-2].unsqueeze(1), R_min, R_max)
            mc_returns_next = norm_rewards(returns.unsqueeze(2)[:,-1].unsqueeze(1), R_min, R_max)
            next_timestep = torch.maximum(r + gamma * q_next, mc_returns_next)
            curr_timestep = torch.maximum(torch.max(q[:, 1:], dim=2, keepdim=True)[0], mc_returns_curr)

            next_dim = torch.cat([curr_timestep, next_timestep], dim=1)
            targets = torch.zeros_like(q)
            target = torch.scatter(targets, 2,  actions[:,-2].unsqueeze(2).long(), next_dim)

            err = loss(q, target.detach())
            optimizer.zero_grad()
            err.backward()
            loss_list.append(err.item())
            optimizer.step()

            if (i+1) % log_loss_steps == 0:
                wandb.log({"train_loss": np.mean(loss_list)})
                loss_list = []

            if (i+1) % eval_steps == 0:
                model.eval()
                wandb.log({"eval_score": eval('HalfCheetah-v4', model, 10)})
                model.train()
            i += 1





if __name__ == "__main__":
    train()