import torch
import torch.nn.functional as F
from pi05.models.gemma import attention

B, H, KV, L, D = 2, 4, 1, 5, 8


def qkv():
    return torch.randn(B, H, L, D), torch.randn(B, KV, L, D), torch.randn(B, KV, L, D)

def test_matches_pytorch_sdpa():
    q, k, v = qkv()
    mask = torch.rand(B, L, L) > 0.3
    mask[..., 0] = True                                      # every query sees at least one key
    out = attention(q, k, v, mask)
    ref = F.scaled_dot_product_attention(                    # PyTorch's own attention as reference
        q, k.expand(-1, H, -1, -1), v.expand(-1, H, -1, -1), attn_mask=mask[:, None]
    ).transpose(1, 2).reshape(B, L, H * D)
    assert out.shape == (B, L, H * D)
    assert torch.allclose(out, ref, atol=1e-5)

def test_masked_keys_have_no_effect():
    q, k, v = qkv()
    mask = torch.ones(B, L, L, dtype=torch.bool)
    mask[..., 3] = False                                     # no query may see key 3
    k2, v2 = k.clone(), v.clone()
    k2[:, :, 3] = 100.0                                      # change key 3 drastically
    v2[:, :, 3] = 100.0
    assert torch.allclose(attention(q, k, v, mask), attention(q, k2, v2, mask), atol=1e-6)

def test_fully_masked_row_is_finite():
    q, k, v = qkv()
    mask = torch.ones(B, L, L, dtype=torch.bool)
    mask[:, 2, :] = False                                    # query 2 sees nothing (like padding)
    assert torch.isfinite(attention(q, k, v, mask)).all()