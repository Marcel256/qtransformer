from single_qtransformer import QTransformer
import torch
from seq_env_wrapper import SequenceEnvironmentWrapper
import numpy as np
import gymnasium as gym


def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()


def play_episode(env: SequenceEnvironmentWrapper, model):
    history = env.reset()
    done = False
    steps = 0
    episode_return = 0
    while not done:
        print(model(transform_state(history)))
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
    return scores



model = QTransformer(4, 2, 128, 4, 3)
checkpoint = torch.load('models_single/model-8.pt')

model.load_state_dict(checkpoint['model_state'])
model.eval()

scores = eval(SequenceEnvironmentWrapper(gym.make('CartPole-v1', render_mode='human'), num_stack_frames=4), model, episodes=1)
print(np.max(scores))
print(np.min(scores))
print(np.mean(scores))