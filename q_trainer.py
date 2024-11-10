import torch
from torch.nn import MSELoss

from qtransformer import QTransformer
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader


import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from d4rl_evaluator import load_d4rl_env, batched_eval

from util import soft_update
from schedulers import get_cosine_schedule_with_warmup

from collections import deque


from wandb_logger import WandbLogger
import dotenv
import os

def norm_rewards(r, R_min, R_max):
    return (r - R_min) / (R_max - R_min)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:
    dotenv.load_dotenv()
    model_config = cfg['model']
    seq_len = model_config['seq_len']
    dataset_file = cfg['dataset']
    train_config = cfg['train']


    batch_size = train_config['batch_size']
    env_config = cfg['env']
    env_name = env_config['id']
    action_dim = env_config['action_dim']
    action_bins = model_config['action_bins']
    use_dueling_head = model_config['dueling']
    use_mc_returns = train_config['mc_returns']
    total_steps = train_config['total_steps']
    weight_decay = train_config['weight_decay']

    discrete_actions = env_config['discrete_actions']
    state_dim = env_config['state_dim']
    hidden_dim = model_config['hidden_dim']

    gamma = cfg['gamma']
    tau = train_config['tau']
    reg_weight = train_config['reg_weight']
    grad_norm = train_config['grad_norm']
    seed = train_config['seed'] if 'seed' in train_config else 0
    model_folder = train_config['model_folder']

    a_min = env_config['action_min']
    a_max = env_config['action_max']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)
    print('device: ', device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    offline_env, data = load_d4rl_env(env_name)

    dataset = SequenceDataset.from_d4rl(data, seq_len, action_bins, gamma)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    R_min = np.min(dataset.returns)
    R_max = np.max(dataset.returns)

    if discrete_actions:
        action_transform = lambda x: x[0]
    else:
        action_transform = lambda x: (x/action_bins) * (a_max - a_min) + a_min
    #env = SequenceEnvironmentWrapper(offline_env, num_stack_frames=seq_len, action_dim=action_dim, action_transform=action_transform)
    model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, dueling=use_dueling_head, device=device)
    target_model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, dueling=use_dueling_head, device=device)
    target_model.eval()
    target_model.to(device)
    model.to(device)
    orig_model = model
    target_model = torch.compile(target_model)
    model = torch.compile(model)
    soft_update(model, target_model, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps*0.1), total_steps)
    loss = MSELoss()

    best_score = -9999
    log_loss_steps = 50
    eval_steps = 1000
    i = 0
    loss_list = deque(maxlen=50)
    td_loss_list = deque(maxlen=50)
    reg_loss_list = deque(maxlen=50)
    eval_episodes=10

    logger = WandbLogger(os.environ['WANDB_ENTITY'], os.environ['WANDB_PROJECT'], cfg)
    print(OmegaConf.to_yaml(cfg))

    # print(batched_eval(env_name, model,eval_episodes, num_stack_frames=seq_len, action_dim=action_dim,action_transform=action_transform))
    while i < total_steps:
        for batch in dataloader:
            states, actions, rewards, returns, terminal, timesteps  = batch

            states = states.float().to(device)
            actions = torch.reshape(actions, (actions.shape[0], seq_len+1, -1)).int().to(device)
            returns = returns.float().to(device)
            rewards = rewards.float().to(device)
            timesteps = timesteps.int().to(device)
            terminal = terminal.unsqueeze(2).float().to(device)
            with torch.no_grad():
                q_next = torch.max(torch.sigmoid(target_model(states[:, 1:], actions[:,-1], timesteps[:,1:])[:, 0].unsqueeze(1)), dim=2, keepdim=True)[0]

            q = torch.sigmoid(model(states[:, :-1], actions[:,-2], timesteps[:,:-1]))

            r = rewards.unsqueeze(2)[:,-2].unsqueeze(1) / (R_max - R_min)

            mc_returns_next = norm_rewards(returns.unsqueeze(2)[:,-1].unsqueeze(1), R_min, R_max)
            q_target_next = r + gamma * (1-terminal[:,-2].unsqueeze(1))*q_next
            if use_mc_returns:
                next_timestep = torch.maximum(q_target_next, mc_returns_next)
            else:
                next_timestep = q_target_next

            if action_dim > 1:
                mc_returns_curr = norm_rewards(returns.unsqueeze(2)[:, -2].unsqueeze(1), R_min, R_max)
                q_target_curr = torch.max(q[:, 1:], dim=2, keepdim=True)[0]
                if use_mc_returns:
                    curr_timestep = torch.maximum(q_target_curr, mc_returns_curr)
                else:
                    curr_timestep = q_target_curr
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
            loss_list.append(err.item())
            td_loss_list.append(td_loss.item())
            reg_loss_list.append(reg_loss.item())
            optimizer.step()
            soft_update(model, target_model, tau)
            scheduler.step()
            if (i+1) % log_loss_steps == 0:
                logger.log({"train_loss": np.mean(loss_list),
                           "td_loss": np.mean(td_loss_list),
                           "reg_loss": np.mean(reg_loss_list)})


            if (i+1) % eval_steps == 0:
                model.eval()
                score = batched_eval(env_name, model, eval_episodes, num_stack_frames=seq_len, action_dim=action_dim,action_transform=action_transform)
                logger.log({"eval_score": offline_env.get_normalized_score(score)})
                if score > best_score:
                    torch.save({'model_state': orig_model.state_dict()}, os.path.join(model_folder, 'best.pt'))
                    best_score = score
                model.train()
            i += 1
    model.eval()
    score = batched_eval(env_name, model, eval_episodes, num_stack_frames=seq_len, action_dim=action_dim,action_transform=action_transform)
    logger.log({"eval_score": offline_env.get_normalized_score(score)})
    torch.save({'model_state': orig_model.state_dict()}, os.path.join(model_folder,'final.pt'))





if __name__ == "__main__":
    train()