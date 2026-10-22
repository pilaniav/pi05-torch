import pytest
import torch
from pi05.models.gemma import GemmaModel
from pi05.models.gemma_config import GemmaConfig, get_gemma_config

VLM = GemmaConfig(width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8, vocab_size=100)
EXPERT = GemmaConfig(width=16, depth=2, mlp_dim=32, num_heads=4, num_kv_heads=1, head_dim=8,
                     vocab_size=None, use_adarms=True)
B, L = 2, 5

@pytest.mark.parametrize("cfg", [VLM, EXPERT], ids=["VLM", "EXPERT"])
def test_forward_shape(cfg):
    model = GemmaModel(cfg)
    embs = torch.randn(B, L, cfg.width)
    positions = torch.arange(L).expand(B, L)
    mask = torch.ones(B, L, L, dtype=torch.bool)
    cond = torch.randn(B, cfg.width) if cfg.use_adarms else None
    assert model(embs, positions, mask, cond).shape == (B, L, cfg.width)

def test_embed_scales_by_sqrt_width():
    model = GemmaModel(VLM)
    ids = torch.tensor([[1, 5, 7]])
    expected = model.embed_tokens.weight[ids] * VLM.width ** 0.5
    assert torch.allclose(model.embed(ids), expected)
    with pytest.raises(ValueError):
        GemmaModel(EXPERT).embed(ids)  

def test_gemma_2b_size_matches_published():
    with torch.device("meta"):                # build the full model without allocating memory
        model = GemmaModel(get_gemma_config("gemma_2b"))
    non_embedding = sum(p.numel() for name, p in model.named_parameters()
                        if not name.startswith("embed_tokens"))
    assert non_embedding == 1_981_884_416     # Gemma 2B total 2,506,172,416 minus its 256000x2048 table