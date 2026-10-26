from dataclasses import dataclass

@dataclass(frozen=True)
class Pi05Config:
    vision_variant: str = "so400m"           # SigLIP variant
    vlm_variant: str = "gemma_2b"            # Paligemma's language model
    expert_variant: str = "gemma_300m"       # action expert
    action_dim: int = 32                     # actions padded to this many dims
    action_horizon: int = 50                 # actions predicted per chunk
    max_token_len: int = 200                 # language tokens (prompt + state), padded
    num_denoise_steps: int = 10              # steps in sample_actions
    time_min_period: float = 4e-3            # sinusoidal time embedding: shortest period
    time_max_period: float = 4.0             # longest period