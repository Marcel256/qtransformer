import torch
from torch.nn import MSELoss
from transformers import DecisionTransformerModel

from transformers.models.decision_transformer import DecisionTransformerConfig
from seq_env_wrapper import SequenceEnvironmentWrapper
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader


import hydra
from omegaconf import DictConfig, OmegaConf

import numpy as np

from d4rl_evaluator import eval, load_d4rl_env

from schedulers import get_cosine_schedule_with_warmup

from collections import deque

from train_logger import NoLogger
from wandb_logger import WandbLogger

def norm_rewards(r, R_min, R_max):
    return (r - R_min) / (R_max - R_min)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg : DictConfig) -> None:

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

    a_min = env_config['action_min']
    a_max = env_config['action_max']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    offline_env, data = load_d4rl_env(env_name)

    dataset = SequenceDataset.from_d4rl(data, seq_len-1, action_bins, gamma)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    R_min = np.min(dataset.returns)
    R_max = np.max(dataset.returns)

    if discrete_actions:
        action_transform = lambda x: x[0]
    else:
        action_transform = lambda x: (x/action_bins) * (a_max - a_min) + a_min
    env = SequenceEnvironmentWrapper(offline_env, num_stack_frames=seq_len, action_dim=action_dim, action_transform=action_transform)
    config = DecisionTransformerConfig(state_dim=state_dim, action_dim=action_dim)
    model = DecisionTransformerModel(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, total_steps, total_steps)
    loss = MSELoss()

    best_score = -9999
    log_loss_steps = 50
    eval_steps = 1000
    i = 0
    loss_list = deque(maxlen=50)
    td_loss_list = deque(maxlen=50)
    reg_loss_list = deque(maxlen=50)

    logger = NoLogger()#WandbLogger(env_config['entity'], env_config['project'])
    print(OmegaConf.to_yaml(cfg))
    #print(eval(env, model, 10))
    while i < total_steps:
        for batch in dataloader:
            states, actions, rewards, returns, terminal = batch

            states = states.float().to(device)
            actions = torch.reshape(actions, (actions.shape[0], seq_len, -1)).int().to(device)
            returns = returns.float().to(device)
            rewards = rewards.float().to(device)
            #terminal = terminal.unsqueeze(2).float().to(device)
            # DT
            outputs = model.forward(states=states, actions=actions, rewards=rewards, returns_to_go=returns)
            err = loss(outputs[1], actions)
            optimizer.zero_grad()
            err.backward()
            optimizer.step()
            scheduler.step()
            if (i+1) % log_loss_steps == 0:
                logger.log({"train_loss": np.mean(loss_list),
                           "td_loss": np.mean(td_loss_list),
                           "reg_loss": np.mean(reg_loss_list)})


            if (i+1) % eval_steps == 0:
                model.eval()
                score = offline_env.get_normalized_score(eval(env, model, 10))
                logger.log({"eval_score": score})
                if score > best_score:
                    torch.save({'model_state': model.state_dict()}, 'models/best.pt')
                    best_score = score
                model.train()
            i += 1
    model.eval()
    score = offline_env.get_normalized_score(eval(env, model, 10))
    logger.log({"eval_score": score})
    torch.save({'model_state': model.state_dict()}, 'models/final.pt')





if __name__ == "__main__":
    train()