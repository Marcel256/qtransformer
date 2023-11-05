from seq_env_wrapper import SequenceEnvironmentWrapper
from qtransformer import QTransformer

import gymnasium as gym
import numpy as np
import torch


def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()

def eval(env_name, model, episodes):
    env = SequenceEnvironmentWrapper(gym.make(env_name), num_stack_frames=4, action_dim=6)

    returns = []
    for i in range(episodes):
        returns.append(play_episode(env, model))

    return np.mean(returns)


def play_episode(env: SequenceEnvironmentWrapper, model):
    env.reset()
    history = env.reset()
    done = False
    steps = 0
    episode_return = 0
    action_bins = 256
    a_min = -1
    a_max = 1
    while not done:
        action = model.predict_action(transform_state(history))[0]
        action = action / action_bins * (a_max-a_min) + a_min
        history, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        steps += 1
        episode_return += reward

    return episode_return



model = QTransformer(17, 6, 256, 256, 4)

#eval('HalfCheetah-v4', model, 1)