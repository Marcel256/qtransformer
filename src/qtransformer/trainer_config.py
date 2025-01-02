from dataclasses import dataclass, field
from qtransformer.model.qtransformer_config import ModelConfig

@dataclass
class EnvType:
    D4RL = "D4RL"
    ATARI = "atari"

@dataclass
class EnvConfig:
    id: str = ""
    discrete_actions: bool = False
    state_dim: int = 0
    action_dim: int = 0
    action_min: float = -1
    action_max: float = 1
    type: EnvType = EnvType.D4RL





@dataclass
class TrainStrategy:
    BC = "BC"
    Q = "Q"

@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    reg_weight: float = 1
    tau: float = 1e-3
    output_dir: int = "models"
    mc_returns: bool = True
    max_grad_norm: int = 1
    gamma: float = 0.99
    strategy: TrainStrategy = TrainStrategy.Q
    train_steps: int = 100000
    R_min: float = 0
    R_max: float = 1


@dataclass
class TrainerConfig:
    env_config: EnvConfig = EnvConfig()
    train_config: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    dataset: str = None
    info: str = ""
    seed: int = 0