import torch
from torch.nn import MSELoss

from qtransformer import QTransformer
from seq_env_wrapper import SequenceEnvironmentWrapper
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader


import hydra
from omegaconf import DictConfig

import numpy as np

from d4rl_evaluator import eval

from util import soft_update

from collections import deque


from wandb_logger import WandbLogger

def norm_rewards(r, R_min, R_max):
    return (r - R_min) / (R_max - R_min)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:

    model_config = cfg['model']
    seq_len = model_config['seq_len']
    dataset_file = cfg['dataset']
    train_config = cfg['train']

    dataset = SequenceDataset(dataset_file, seq_len)
    batch_size = train_config['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    env_config = cfg['env']
    env_name = env_config['id']
    action_dim = env_config['action_dim']
    action_bins = model_config['action_bins']

    discrete_actions = env_config['discrete_actions']
    state_dim = env_config['state_dim']
    hidden_dim = model_config['hidden_dim']

    R_min = env_config['R_min']
    R_max = env_config['R_max']
    gamma = cfg['gamma']
    tau = train_config['tau']
    reg_weight = train_config['reg_weight']

    a_min = env_config['action_min']
    a_max = env_config['action_max']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if discrete_actions:
        action_transform = lambda x: x[0]
    else:
        action_transform = lambda x: (x/action_bins) * (a_max - a_min) + a_min
    env = SequenceEnvironmentWrapper(gym.make(env_name), num_stack_frames=seq_len, action_dim=action_dim, action_transform=action_transform)
    model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, device=device)
    target_model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, device=device)
    target_model.eval()
    target_model.to(device)
    model.to(device)
    soft_update(model, target_model, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])
    loss = MSELoss()

    log_loss_steps = 50
    eval_steps = 500
    i = 0
    epochs = train_config['epochs']
    loss_list = deque(maxlen=50)
    td_loss_list = deque(maxlen=50)
    reg_loss_list = deque(maxlen=50)

    logger = WandbLogger(env_config['entity'], env_config['project'])
    print(eval(env, model, 10))
    for epoch in range(epochs):
        for batch in dataloader:
            states, actions, rewards, returns, terminal = batch

            states = states.float().to(device)
            actions = torch.reshape(actions, (actions.shape[0], seq_len+1, -1)).int().to(device)
            returns = returns.float().to(device)
            rewards = rewards.float().to(device)
            terminal = terminal.unsqueeze(2).float().to(device)
            with torch.no_grad():
                q_next = torch.max(torch.sigmoid(target_model(states[:, 1:], actions[:,-1])[:, 0].unsqueeze(1)), dim=2, keepdim=True)[0]

            q = torch.sigmoid(model(states[:, :-1], actions[:,-2]))

            r = rewards.unsqueeze(2)[:,-2].unsqueeze(1) / (R_max - R_min)

            mc_returns_next = norm_rewards(returns.unsqueeze(2)[:,-1].unsqueeze(1), R_min, R_max)
            next_timestep = torch.maximum(r + gamma * (1-terminal[:,-2].unsqueeze(1))*q_next, mc_returns_next)

            if action_dim > 1:
                mc_returns_curr = norm_rewards(returns.unsqueeze(2)[:, -2].unsqueeze(1), R_min, R_max)
                curr_timestep = torch.maximum(torch.max(q[:, 1:], dim=2, keepdim=True)[0], mc_returns_curr)
                next_dim = torch.cat([curr_timestep, next_timestep], dim=1)
            else:
                next_dim = next_timestep

            pred = torch.gather(q, 2, actions[:,-2].unsqueeze(2).long())

            action_mask = torch.ones_like(q)
            action_mask.scatter_(2, actions[:,-2].unsqueeze(2).long(), 0)

            bin_sum = torch.sum( (q**2) * action_mask)

            reg_loss = bin_sum / action_mask.sum()

            td_loss = loss(pred, next_dim.detach())

            err = (td_loss + reg_weight * reg_loss)/2
            optimizer.zero_grad()
            err.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            loss_list.append(err.item())
            td_loss_list.append(td_loss.item())
            reg_loss_list.append(reg_loss.item())
            optimizer.step()
            soft_update(model, target_model, tau)
            if (i+1) % log_loss_steps == 0:
                logger.log({"train_loss": np.mean(loss_list),
                           "td_loss": np.mean(td_loss_list),
                           "reg_loss": np.mean(reg_loss_list)})


            if (i+1) % eval_steps == 0:
                model.eval()
                logger.log({"eval_score": eval(env, model, 5)})
                torch.save({'model_state': model.state_dict()}, 'models/model-{}.pt'.format(i))
                model.train()
            i += 1





if __name__ == "__main__":
    train()