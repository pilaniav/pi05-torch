import torch
import torch.nn.functional as F
from pi05.models.gemma import GemmaMLP

def test_shape_preserved():
    mlp = GemmaMLP(width=16, mlp_dim=64)
    assert mlp(torch.randn(2, 5, 16)).shape == (2, 5, 16)

def test_matches_formula():
    mlp = GemmaMLP(width=16, mlp_dim=64)
    x = torch.randn(2, 5, 16)
    g, u, d = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    expected = (F.gelu(x @ g.T, approximate="tanh") * (x @ u.T)) @ d.T
    assert torch.allclose(mlp(x), expected, atol=1e-5)

def test_tokens_processed_independently():
    mlp = GemmaMLP(width=16, mlp_dim=64)
    x = torch.randn(1, 5, 16)
    full = mlp(x)
    one = mlp(x[:, 2:3])            # run token 2 alone
    assert torch.allclose(full[:, 2:3], one, atol=1e-6)