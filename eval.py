from seq_env_wrapper import SequenceEnvironmentWrapper
from qtransformer import QTransformer

import gymnasium as gym
import numpy as np
import torch


def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()

def eval(env, model, episodes):


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
    while not done:
        action = model.predict_action(transform_state(history))[0]
        history, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        steps += 1
        episode_return += reward
        if steps % 100 == 0:
            print(steps)

    return episode_return