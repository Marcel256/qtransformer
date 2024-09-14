import gym
import d4rl

import torch

import numpy as np
from seq_env_wrapper import SequenceEnvironmentWrapper


def load_d4rl_env(env_name):
    env = gym.make(env_name)
    return env, env.get_dataset()



def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()

def eval(env, model, episodes):


    returns = []
    for i in range(episodes):
        returns.append(play_episode(env, model))

    return np.mean(returns)


def play_episode(env: SequenceEnvironmentWrapper, model):
    history = env.reset()
    done = False
    steps = 0
    episode_return = 0
    while not done:
        state = transform_state(history)
        action = model.predict_action(state, torch.clip(torch.arange(steps-state.shape[1], steps),0, 1000).unsqueeze(0))[0]
        history, reward, done, info = env.step(action)
        steps += 1
        episode_return += reward

    return env.get_normalized_score(episode_return)



def transform_batched(history):
    states = [h['observations'] for h in history]
    return torch.from_numpy(np.stack(states)).float()
def batched_eval(env_name, model, episodes, num_stack_frames=4, action_dim=1, action_transform=None):
    envs = []
    returns = []
    history = []
    dones = []
    for ep in range(episodes):
        env = SequenceEnvironmentWrapper(gym.make(env_name), num_stack_frames=num_stack_frames, action_dim=action_dim, action_transform=action_transform)
        history.append(env.reset())
        returns.append(0)
        dones.append(False)
        envs.append(env)
    steps = 0
    while not all(dones):
        states = transform_batched(history)
        timesteps = torch.clip(torch.arange(steps - states.shape[1], steps), 0, 1000).unsqueeze(0)
        actions = model.predict_action(states, timesteps)
        for i in range(episodes):
            if not dones[i]:
                next_state, reward, done, info = envs[i].step(actions[i])
                history[i] = next_state
                returns[i] += reward
                dones[i] = done

        steps += 1

    return returns




