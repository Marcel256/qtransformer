import torch

from qtransformer import QTransformer
from seq_env_wrapper import SequenceEnvironmentWrapper

import hydra
from omegaconf import DictConfig, OmegaConf
from d4rl_evaluator import eval, load_d4rl_env


def norm_rewards(r, R_min, R_max):
    return (r - R_min) / (R_max - R_min)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    model_config = cfg['model']
    seq_len = model_config['seq_len']

    env_config = cfg['env']
    env_name = env_config['id']
    action_dim = env_config['action_dim']
    action_bins = model_config['action_bins']
    use_dueling_head = model_config['dueling']

    discrete_actions = env_config['discrete_actions']
    state_dim = env_config['state_dim']
    hidden_dim = model_config['hidden_dim']

    a_min = env_config['action_min']
    a_max = env_config['action_max']

    checkpoint = cfg['checkpoint']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    offline_env, data = load_d4rl_env(env_name)
    if discrete_actions:
        action_transform = lambda x: x[0]
    else:
        action_transform = lambda x: (x / action_bins) * (a_max - a_min) + a_min
    env = SequenceEnvironmentWrapper(offline_env, num_stack_frames=seq_len, action_dim=action_dim,
                                     action_transform=action_transform)
    model = QTransformer(state_dim, action_dim, hidden_dim, action_bins, seq_len, dueling=use_dueling_head,
                         device=device)

    print('Loading ', checkpoint)
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    print(OmegaConf.to_yaml(cfg))
    print('Normalized Score: ', offline_env.get_normalized_score(eval(env, model, 100)))


if __name__ == "__main__":
    eval()