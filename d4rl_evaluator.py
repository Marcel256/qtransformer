import gym

import torch

import numpy as np
from seq_env_wrapper import SequenceEnvironmentWrapper



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
        history, reward, done, info = env.step(action)
        steps += 1
        episode_return += reward

    return episode_return