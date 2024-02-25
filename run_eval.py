from seq_env_wrapper import SequenceEnvironmentWrapper
import torch
from qtransformer import QTransformer

import gymnasium as gym

def transform_state(history):
    return torch.from_numpy(history['observations']).unsqueeze(0).float()


def play_episode(env: SequenceEnvironmentWrapper, model):
    history = env.reset()
    done = False
    steps = 0
    episode_return = 0
    action_bins = 256
    a_min = -1
    a_max = 1
    while not done:
        print('Hier')
        action = model.predict_action(transform_state(history))[0]
        action = action / action_bins * (a_max-a_min) + a_min
        history, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        steps += 1
        episode_return += reward
        break

    return episode_return


model = QTransformer(17, 6, 256, 256, 4)
checkpoint = torch.load('models/model-999.pt')

model.load_state_dict(checkpoint['model_state'])
model.eval()

play_episode(SequenceEnvironmentWrapper(gym.make('HalfCheetah-v4'), num_stack_frames=4, action_dim=6), model)

