import pytest
import torch
from pi05.models.gemma import GemmaBlock, rope_cos_sin
from pi05.models.gemma_config import GemmaConfig

VLM = GemmaConfig(width=32, depth=1, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8)
EXPERT = GemmaConfig(width=16, depth=1, mlp_dim=32, num_heads=4, num_kv_heads=1, head_dim=8,
                     use_adarms=True)
B, L = 2, 5

def inputs(cfg):
    x = torch.randn(B, L, cfg.width)
    cos, sin = rope_cos_sin(torch.arange(L).expand(B, L), cfg.head_dim, cfg.rope_theta)
    mask = torch.ones(B, L, L, dtype=torch.bool)
    cond = torch.randn(B, cfg.width) if cfg.use_adarms else None
    return x, cos, sin, mask, cond

@pytest.mark.parametrize("cfg", [VLM, EXPERT], ids=["VLM", "EXPERT"])
def test_shape(cfg):
    x, cos, sin, mask, cond = inputs(cfg)
    assert GemmaBlock(cfg)(x, cos, sin, mask, cond).shape == x.shape

def test_expert_block_is_identity_at_init():
    # adaRMS dense layers start at zero -> both gates are 0 -> both residual branches vanish
    x, cos, sin, mask, cond = inputs(EXPERT)
    assert torch.allclose(GemmaBlock(EXPERT)(x, cos, sin, mask, cond), x)