import torch
from torch import nn

from .gemma import GemmaModel
from .gemma_config import GemmaConfig
from .siglip import SiglipVision
from .siglip_config import SiglipConfig

class PaliGemma(nn.Module):
    """
    SigLIP vision tower + linear projector + Gemma language model.
    """

    def __init__(self, vision_config: SiglipConfig, llm_config: GemmaConfig):
        super().__init__()
        self.vision = SiglipVision(vision_config)
        self.projector = nn.Linear(vision_config.width, llm_config.width)
        self.llm = GemmaModel(llm_config)

    def embed_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        images (B, 3, 224, 224) in [-1, 1] -> (B, 256, llm width). Not scaled (unlike text).
        """
        return self.projector(self.vision(images))

    def embed_language(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids (B, L) -> (B, L, llm width), scaled by sqrt(width) inside GemmaModel.embed.
        """
        return self.llm.embed(ids)