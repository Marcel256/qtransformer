import minari

import numpy as np
import pickle
dataset = minari.load_dataset('HalfCheetah-v4-random-v0')

result = dict()
observations = []
actions = []
rewards = []
terminal = []
returns = []


N=256
a_min = -1
a_max = 1

gamma = 0.99

#prepend zeros before each episode
padding = 3

for episode in dataset.iterate_episodes():
    observations.append(np.zeros((padding, episode.observations.shape[1])))
    observations.append(episode.observations[:-1])
    a = np.concatenate((np.zeros((padding, episode.actions.shape[1])), episode.actions), axis=0)
    actions.append(((a-a_min)/(a_max - a_min)*N).astype(np.uint8))
    rewards.append(np.zeros((padding,)))
    rewards.append(episode.rewards)
    r = np.concatenate(rewards, axis=0)
    ret = np.zeros_like(r)
    ret[-1] = r[-1]
    for i in reversed(range(r.shape[0]-1)):
        ret[i] = r[i] + gamma*ret[i+1]
    returns.append(ret)
    terminal.append(np.zeros((padding,)))
    terminal.append(np.logical_or(episode.terminations, episode.truncations))


#padding at the end
observations.append(np.zeros((1, observations[0].shape[1])))
actions.append(np.zeros((1, actions[0].shape[1])))
rewards.append(np.zeros((1,)))
returns.append(np.zeros((1,)))
terminal.append(np.zeros((1,)))


result['observations'] = np.concatenate(observations, axis=0)
result['actions'] = np.concatenate(actions, axis=0)
result['rewards'] = np.concatenate(rewards, axis=0)
result['returns'] = np.concatenate(returns, axis=0)
result['terminals'] = np.concatenate(terminal, axis=0)

with open('data/random_seq.pkl', 'wb') as fh:
    pickle.dump(result, fh)