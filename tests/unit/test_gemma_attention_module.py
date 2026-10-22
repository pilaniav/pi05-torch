import pytest
import torch
from pi05.models.gemma import GemmaAttention, rope_cos_sin
from pi05.models.gemma_config import GemmaConfig


VLM = GemmaConfig(width=32, depth=1, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8)
EXPERT = GemmaConfig(width=16, depth=1, mlp_dim=32, num_heads=4, num_kv_heads=1, head_dim=8)
B, L = 2, 5


def inputs(cfg):
    x = torch.randn(B, L, cfg.width)
    cos, sin = rope_cos_sin(torch.arange(L).expand(B, L), cfg.head_dim, cfg.rope_theta)
    return x, cos, sin


@pytest.mark.parametrize("cfg", [VLM, EXPERT], ids=["VLM", "EXPERT"])
def test_shapes(cfg):
    attn = GemmaAttention(cfg)
    x, cos, sin = inputs(cfg)
    mask = torch.ones(B, L, L, dtype=torch.bool)                    # every token sees every token
    q, k, v = attn.project_qkv(x)
    assert q.shape == (B, 4, L, 8) and k.shape == v.shape == (B, 1, L, 8)
    assert attn(x, cos, sin, mask).shape == (B, L, cfg.width)


def test_causal_mask_hides_future_tokens():
    attn = GemmaAttention(VLM)
    x, cos, sin = inputs(VLM)
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool)).expand(B, L, L)   # causal
    x2 = x.clone()
    x2[:, -1] = 100.0                                              # change only the last token
    out1, out2 = attn(x, cos, sin, mask), attn(x2, cos, sin, mask)
    assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)   # earlier tokens unaffected
    assert not torch.allclose(out1[:, -1], out2[:, -1])            # last token did change