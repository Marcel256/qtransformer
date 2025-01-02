import gym
import d4rl

def load_d4rl_dataset(env_name: str):
    env = gym.make(env_name)
    return env.get_dataset()