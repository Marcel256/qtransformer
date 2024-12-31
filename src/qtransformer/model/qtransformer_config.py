from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_layers: int
    n_heads: int
    hidden_dim: int
    seq_len: int
    action_bins: int
    dueling: bool
    max_timestep_emb: int
    conv_encoder: bool