import torch
from torch.nn import MSELoss

from single_qtransformer import QTransformer
from sequence_dataset import SequenceDataset
from torch.utils.data import DataLoader

from torch.nn.utils import clip_grad_norm_

from seq_env_wrapper import SequenceEnvironmentWrapper
from omegaconf import DictConfig

import numpy as np

from eval import eval

import matplotlib.pyplot as plt
import gymnasium as gym
from collections import deque

def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()


def play_episode(env: SequenceEnvironmentWrapper, model):
    history = env.reset()
    done = False
    steps = 0
    episode_return = 0
    while not done:
        action = model.predict_action(transform_state(history))[0]
        history, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        steps += 1
        episode_return += reward

    return episode_return


def eval(env: SequenceEnvironmentWrapper, model, episodes=10):
    model.eval()
    scores = [play_episode(env, model) for ep in range(episodes)]
    model.train()
    return np.mean(scores)

def norm_rewards(r, R_min, R_max):
    return r/R_max


def norm_rewards_2(r, R_min, R_max):
    return (r-R_min)/(R_max-R_min)

def soft_update(local_model, target_model, tau):
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def cql_loss(q_values, current_action):
    """Computes the CQL loss for a batch of Q-values and actions."""
    logsumexp = torch.logsumexp(q_values, dim=1, keepdim=True)
    q_a = q_values.gather(1, current_action)

    return (logsumexp - q_a).mean()

def train() -> None:
    dataset = SequenceDataset('data/pole_random_seq.pkl', 4)

    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
    env = SequenceEnvironmentWrapper(gym.make('CartPole-v1'), num_stack_frames=4)

    gamma = 0.99
    model = QTransformer(4, 2, 128, 4, 3)
    target_model = QTransformer(4, 2, 128, 4, 3)
    target_model.eval()
    soft_update(model, target_model, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = MSELoss()
    model.train()
    i = 0
    epochs = 10
    print("Eval: ", eval(env, model))
    loss_list = deque(maxlen=50)
    for epoch in range(epochs):
        for batch in dataloader:
            states, actions, rewards, returns, terminal = batch

            #returns = returns.float()
            a = actions[:,-2].unsqueeze(1).long()
            not_terminal = (1-terminal[:,-2]).float().unsqueeze(1)
            with torch.no_grad():
                q_next = torch.max(target_model(states[:, 1:].float())[:, 0].unsqueeze(1), dim=-1, keepdim=True)[0]

            q = model(states[:, :-1].float())

            r = rewards[:,-2].float().unsqueeze(1) / 250  #norm_rewards(rewards.float().unsqueeze(2)[:,-2].unsqueeze(1), R_min, R_max)
            ret = returns[:,-2].float().unsqueeze(1) / 250 #norm_rewards(rewards.float().unsqueeze(2)[:,-2].unsqueeze(1), R_min, R_max)

            q_pred = q.gather(1, a)

            action_mask = torch.ones_like(q)
            action_mask.scatter_(1, a, 0)

            reg_loss = torch.sum(((q-0) * action_mask)**2) / action_mask.sum()

            q_target = r + not_terminal * (gamma * q_next)
            q_target = torch.maximum(q_target, ret)

            #cql1_loss = cql_loss(q, a)

            bellman_error = loss(q_pred, q_target)

            #q_loss = cql1_loss + 0.5 * bellman_error

            q_loss = 0.5 * bellman_error + 0.5 * reg_loss
            loss_list.append(q_loss.item())

            if i % 100 == 0:
                print('Loss: ', np.mean(loss_list))

            optimizer.zero_grad()
            q_loss.backward()
            clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            soft_update(model, target_model, 1e-2)
            i += 1
        print("Eval: ", eval(env, model))
        torch.save({'model_state': model.state_dict()}, 'models_single/model-{}.pt'.format(epoch))





if __name__ == "__main__":
    train()