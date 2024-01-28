import minari

import numpy as np
import pickle
dataset = minari.load_dataset('CartPole-v1-random-v0')

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
returns = []

for episode in dataset.iterate_episodes():
    returns.append(np.sum(episode.rewards))

print(np.min(returns))
print(np.max(returns))
print(np.mean(returns))