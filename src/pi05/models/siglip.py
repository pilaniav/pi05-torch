import torch
from torch import nn
from transformers import SiglipVisionConfig, SiglipVisionModel

from .siglip_config import SiglipConfig

class SiglipVision(nn.Module):
    """
    Images (B, 3, H, W) in [-1, 1] -> patch tokens (B, num_patches, width).
    Wraps HF's SiglipVisionModel; the rest of the code only sees this class.
    """

    def __init__(self, config: SiglipConfig):
        super().__init__()
        self.config = config
        self.model = SiglipVisionModel(SiglipVisionConfig(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.width,
            num_hidden_layers=config.depth,
            intermediate_size=config.mlp_dim,
            num_attention_heads=config.num_heads,
            layer_norm_eps=config.layer_norm_eps,
            hidden_act="gelu_pytorch_tanh",          # set explicitly, not left to library defaults
            vision_use_head=False,                   # PaliGemma keeps all patch tokens, no pooling
            attn_implementation=config.attn_implementation
        ))

    @property
    def num_patches(self) -> int:
        return (self.config.image_size // self.config.patch_size) ** 2

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=images).last_hidden_state