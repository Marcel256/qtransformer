from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layers: int = 3
    n_heads: int = 1
    hidden_dim: int = 128
    seq_len: int = 1
    action_bins: int = 128
    dueling: bool = True
    max_timestep_emb: int = 1000
    conv_encoder: bool = False