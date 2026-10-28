"""
Data loading. Placeholder: random batches with the right shapes.
"""

from collections.abc import Iterator

import torch

from pi05.models.gemma_config import get_gemma_config
from pi05.models.observation import Observation
from pi05.models.pi05_config import Pi05Config
from pi05.models.siglip_config import get_siglip_config

NUM_CAMERAS = 3


def fake_batches(
    model_config: Pi05Config, batch_size: int, seed: int
) -> Iterator[tuple[Observation, torch.Tensor]]:
    """
    Endless random (obs, actions) batches. TODO: replace with the real loader.
    """
    generator = torch.Generator().manual_seed(seed)
    image_size = get_siglip_config(model_config.vision_variant).image_size
    vocab_size = get_gemma_config(model_config.vlm_variant).vocab_size
    B, L = batch_size, model_config.max_token_len
    H, D = model_config.action_horizon, model_config.action_dim

    while True:
        images = torch.rand(B, NUM_CAMERAS, 3, image_size, image_size, generator=generator)
        obs = Observation(
            images=images * 2 - 1, # [-1, 1]
            image_masks=torch.ones(B, NUM_CAMERAS, dtype=torch.bool),
            prompt_ids=torch.randint(0, vocab_size, (B, L), generator=generator),
            prompt_mask=torch.ones(B, L, dtype=torch.bool)
        )
        actions = torch.randn(B, H, D, generator=generator)
        yield obs, actions