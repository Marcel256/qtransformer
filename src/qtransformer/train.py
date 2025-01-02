import os
import sys
if not __package__:
    package_source_path = os.path.dirname(os.path.dirname(__file__))
    print(package_source_path)
    sys.path.insert(0, package_source_path)
import hydra
from omegaconf import OmegaConf, open_dict

from qtransformer.evaluator.evaluator import Evaluator
from qtransformer.loss.bc_loss import BCLoss
from qtransformer.loss.q_loss import QLoss
from qtransformer.model.qtransformer import QTransformer
from qtransformer.trainer_config import TrainerConfig, EnvType, TrainStrategy
from qtransformer.trainer import Trainer
from qtransformer.data.sequence_dataset import SequenceDataset

import torch
from torch.utils.data import DataLoader

import numpy as np



@hydra.main(config_path=os.path.join(os.getcwd(), "conf"), config_name="cfg")
def train(trainer_config: TrainerConfig) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = trainer_config.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    discrete_actions = False
    evaluator: Evaluator = None
    if trainer_config.env_config.type == EnvType.D4RL:
        from qtransformer.env.d4rl_utils import load_d4rl_dataset
        from qtransformer.evaluator.d4rl_evaluator import D4RLEvaluator
        evaluator = D4RLEvaluator(trainer_config)
        data = load_d4rl_dataset(trainer_config.env_config.id)
        discrete_actions = False
    elif trainer_config.env_config.type == EnvType.ATARI:
        discrete_actions = True
        from qtransformer.evaluator.atari_evaluator import AtariEvaluator
        from qtransformer.env.atari_utils import load_atari_dataset
        evaluator = AtariEvaluator()
        data = load_atari_dataset(trainer_config.dataset)
        data['observations'] /= 255

    dataset = SequenceDataset.from_d4rl(data, trainer_config.model.seq_len+1, trainer_config.model.action_bins, trainer_config.train_config.gamma, discrete_actions=discrete_actions)
    R_min = np.min(dataset.returns)
    R_max = np.max(dataset.returns)
        
    dataloader = DataLoader(dataset, batch_size=trainer_config.train_config.batch_size, shuffle=True)
    model = QTransformer(trainer_config.env_config.state_dim, trainer_config.env_config.action_dim, trainer_config.model, device)
    model.to(device)
    if trainer_config.train_config.strategy == TrainStrategy.Q:
        loss = QLoss(model, trainer_config.train_config, R_min, R_max)
    elif trainer_config.train_config.strategy == TrainStrategy.BC:
        loss = BCLoss(model)
    trainer = Trainer(model, trainer_config, loss, evaluator, dataloader, device)
    trainer.train()


if __name__ == "__main__":
    train()
