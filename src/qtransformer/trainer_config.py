from dataclasses import dataclass, field
from qtransformer.model.qtransformer_config import ModelConfig

@dataclass
class EnvType:
    D4RL = 0
    ATARI = 1

@dataclass
class EnvConfig:
    id: str
    discrete_actions: bool
    state_dim: int
    action_dim: int
    action_min: float
    action_max: float
    type: EnvType





@dataclass
class TrainStrategy:
    BC = 1
    Q = 2

@dataclass
class TrainConfig:
    lr: float
    batch_size: int
    reg_weight: float
    tau: float
    output_dir: int
    mc_returns: bool
    max_grad_norm: int
    gamma: float
    strategy: TrainStrategy
    train_steps: int
    R_min: float = 0
    R_max: float = 1


@dataclass
class TrainerConfig:
    env_config: EnvConfig = field(default_factoy=EnvConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: str = None
    info: str = ""