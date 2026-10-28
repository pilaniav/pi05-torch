from dataclasses import dataclass

import torch

@dataclass
class Observation:
    """
    What pi0.5 receives per batch. State is already inside the prompt (discretized text).
    """
    images: torch.Tensor        # (B, N_cams, 3, H, W) float in [-1, 1], fixed camera order
    image_masks: torch.Tensor   # (B, N_cams) bool: False = camera missing for that sample
    prompt_ids: torch.Tensor    # (B, L) int: tokenized task text + discretized state
    prompt_mask: torch.Tensor   # (B, L) bool: False = padding

    def to(self, device: torch.device) -> "Observation":
        return Observation(
            images=self.images.to(device),
            image_masks=self.image_masks.to(device),
            prompt_ids=self.prompt_ids.to(device),
            prompt_mask=self.prompt_mask.to(device),
        )