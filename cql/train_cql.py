import gym
from collections import deque
import torch
from cql.cql_agent import CQLAgent
from qtransformer.data.sequence_dataset import SequenceDataset

from torch.utils.data import DataLoader


def eval(env, agent, episodes=20):
    ret = 0
    for i in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.get_action(obs, 0)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            if terminated or truncated:
                done = True

    return ret/episodes


def train():
    env = gym.make('CartPole-v1')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    eps = 1.
    steps = 0
    average10 = deque(maxlen=10)
    total_steps = 0

    agent = CQLAgent(state_size=env.observation_space.shape,
                         action_size=env.action_space.n,
                         device=device)

    dataset = SequenceDataset('data/pole_random_seq.pkl', 1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    print('Eval: ', eval(env, agent, episodes=100))
    for epoch in range(30):
        for batch in dataloader:
            obs, action, rewards, returns, terminal = batch
            agent.learn(obs[:,0,:].float(), action[:,0].long().unsqueeze(1), rewards[:,0].float().unsqueeze(1), obs[:,1,:].float(), terminal[:,0].float().unsqueeze(1))
        print('Eval: ', eval(env, agent))
        #print("Episode: {} | Reward: {} | Q Loss: {} | Steps: {}".format(i, rewards, loss, steps, ))




if __name__ == "__main__":
    train()