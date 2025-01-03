import torch
from torch.nn import MSELoss

from qtransformer import QTransformer
from seq_env_wrapper import SequenceEnvironmentWrapper
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader


import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from d4rl_evaluator import eval, load_d4rl_env

from util import soft_update

from collections import deque


from wandb_logger import WandbLogger


def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()

def gather_data(env, model, episodes):
    states = []
    actions = []
    rewards = []
    terminals = []
    returns = []

    for episode in range(episodes):
        s, a, r, t, ret = play_episode(env, model)
        states.extend(s)
        actions.extend(a)
        rewards.extend(r)
        terminals.extend(t)
        returns.append(ret)

    return {'observations': np.stack(states, axis=0),
            'actions': np.stack(actions, axis=0),
            'rewards': np.stack(rewards, axis=0),
            'terminals': np.stack(terminals, axis=0)}

def play_episode(env: SequenceEnvironmentWrapper, model):
    env.reset()
    history = env.reset()
    done = False
    steps = 0
    episode_return = 0
    states = []
    actions = []
    rewards = []
    terminals = []
    while not done:
        action = model.predict_action(transform_state(history))[0]
        history, reward, done, info = env.step(action)
        states.append(history['observations'][-1])
        actions.append(action)
        rewards.append(reward)
        terminals.append(done)
        steps += 1
        episode_return += reward

    return states, actions, rewards, terminals, episode_return

def norm_rewards(r, R_min, R_max):
    return (r - R_min) / (R_max - R_min)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:

    model_config = cfg['model']
    seq_len = model_config['seq_len']
    train_config = cfg['train']


    batch_size = train_config['batch_size']
    env_config = cfg['env']
    env_name = env_config['id']
    action_dim = env_config['action_dim']
    action_bins = model_config['action_bins']
    use_dueling_head = model_config['dueling']
    use_mc_returns = train_config['mc_returns']

    discrete_actions = env_config['discrete_actions']
    state_dim = env_config['state_dim']
    hidden_dim = model_config['hidden_dim']

    gamma = cfg['gamma']
    tau = train_config['tau']
    reg_weight = train_config['reg_weight']
    grad_norm = train_config['grad_norm']

    a_min = env_config['action_min']
    a_max = env_config['action_max']
    checkpoint = cfg['checkpoint']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    offline_env, data = load_d4rl_env(env_name)

    # Load dataset to compute R_max / R_min
    dataset = SequenceDataset.from_d4rl(data, seq_len, action_bins, gamma)

    R_min = np.min(dataset.returns)
    R_max = np.max(dataset.returns)


    if discrete_actions:
        action_transform = lambda x: x[0]
    else:
        action_transform = lambda x: (x/action_bins) * (a_max - a_min) + a_min
    env = SequenceEnvironmentWrapper(offline_env, num_stack_frames=seq_len, action_dim=action_dim, action_transform=action_transform)
    model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, dueling=use_dueling_head, device=device)
    target_model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, dueling=use_dueling_head, device=device)

    print('Finetuning From ', checkpoint)
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    soft_update(model, target_model, 1)
    target_model.eval()
    target_model.to(device)


    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'])
    loss = MSELoss()

    best_score = -9999
    log_loss_steps = 50
    eval_steps = 1500
    i = 0
    epochs = train_config['epochs']
    loss_list = deque(maxlen=50)
    td_loss_list = deque(maxlen=50)
    reg_loss_list = deque(maxlen=50)

    logger = WandbLogger(env_config['entity'], env_config['project'])
    print(OmegaConf.to_yaml(cfg))
    for epoch in range(epochs):
        model.eval()
        data = gather_data(env, model, 100)
        dataset = SequenceDataset.from_rollouts(data, seq_len, gamma)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model.train()
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
            if (i+1) % log_loss_steps == 0:
                logger.log({"train_loss": np.mean(loss_list),
                           "td_loss": np.mean(td_loss_list),
                           "reg_loss": np.mean(reg_loss_list)})


            if (i+1) % eval_steps == 0:
                model.eval()
                score = offline_env.get_normalized_score(eval(env, model, 10))
                logger.log({"eval_score": score})
                if score > best_score:
                    torch.save({'model_state': model.state_dict()}, 'models/finetuned.pt')
                    best_score = score
                model.train()
            i += 1





if __name__ == "__main__":
    train()