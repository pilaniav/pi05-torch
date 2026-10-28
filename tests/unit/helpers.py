"""
Shared test helpers: tiny Pi05 config, fake observation, nonzero adaRMS gates.
"""

import torch

from pi05.models.gemma import RMSNorm
from pi05.models.observation import Observation
from pi05.models.pi05_config import Pi05Config

TINY = Pi05Config(
    vision_variant="siglip_tiny",
    vlm_variant="gemma_tiny",
    expert_variant="gemma_tiny_expert",
    action_dim=6,
    action_horizon=4,
)


def make_obs(B=2):
    return Observation(
        images=torch.randn(B, 3, 3, 28, 28),
        image_masks=torch.ones(B, 3, dtype=torch.bool),
        prompt_ids=torch.randint(0, 100, (B, 5)),
        prompt_mask=torch.ones(B, 5, dtype=torch.bool),
    )


def randomize_adarms_gates(model):
    """
    adaRMS gates are zero at init, which would hide bugs in the expert.
    """
    for m in model.modules():
        if isinstance(m, RMSNorm) and m.adaptive:
            torch.nn.init.normal_(m.dense.weight, std=0.5)