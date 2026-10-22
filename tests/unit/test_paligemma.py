import torch
from pi05.models.gemma_config import GemmaConfig
from pi05.models.paligemma import PaliGemma
from pi05.models.siglip import SiglipVision
from pi05.models.siglip_config import SiglipConfig

VISION = SiglipConfig(image_size=28, patch_size=14, width=24, depth=2, mlp_dim=48, num_heads=4)  # 2x2 = 4 patches
LLM = GemmaConfig(width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8, vocab_size=100)

def test_siglip_shape_and_no_pooling_head():
    vision = SiglipVision(VISION)
    out = vision(torch.randn(2, 3, 28, 28))
    assert out.shape == (2, vision.num_patches, 24) and vision.num_patches == 4
    assert not any("head" in name for name, _ in vision.named_parameters())
    assert vision.model.config._attn_implementation == "eager"

def test_paligemma_embeddings_have_llm_width():
    model = PaliGemma(VISION, LLM)
    assert model.embed_image(torch.randn(2, 3, 28, 28)).shape == (2, 4, 32)
    assert model.embed_language(torch.tensor([[1, 2, 3]])).shape == (1, 3, 32)