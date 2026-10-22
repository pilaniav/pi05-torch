import torch
from pi05.models.gemma import apply_rope, rope_cos_sin

D = 8            # tiny head_dim -> 4 pairs: (0,4), (1,5), (2,6), (3,7)
THETA = 10_000.0


def rope(x, pos):
    """
    x  : (1, H, L, D)
    pos: (L,)
    """
    cos, sin = rope_cos_sin(torch.tensor([pos]), D, THETA)
    return apply_rope(x, cos, sin)

def test_cos_sin_matches_formula():
    cos, sin = rope_cos_sin(torch.tensor([[5]]), D, THETA)
    angles = 5 / THETA ** (torch.arange(D // 2) / (D // 2))    
    expected = torch.cat([angles, angles])                    
    assert torch.allclose(cos[0, 0], expected.cos())
    assert torch.allclose(sin[0, 0], expected.sin())

def test_pairs_first_half_with_second_half():
    x = torch.zeros(1, 1, 1, D)
    x[..., 0] = 1.0                                          
    out = rope(x, [1])                                         
    assert torch.isclose(out[0, 0, 0, 0], torch.cos(torch.tensor(1.0)))
    assert torch.isclose(out[0, 0, 0, 4], torch.sin(torch.tensor(1.0)))

def test_score_depends_only_on_relative_position():
    q, k = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)
    s1 = (rope(q, [3]) * rope(k, [5])).sum()                   # distance 2
    s2 = (rope(q, [10]) * rope(k, [12])).sum()                 # distance 2, shifted
    assert torch.allclose(s1, s2, atol=1e-5)

def test_rotation_keeps_shape_length_and_dtype():
    x = torch.randn(1, 4, 3, D)
    out = rope(x, [0, 5, 17])
    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)
    assert rope(x.bfloat16(), [0, 5, 17]).dtype == torch.bfloat16
