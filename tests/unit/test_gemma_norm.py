import pytest
import torch
from pi05.models.gemma import RMSNorm

"""
Run these from pi05-torch/

add pytest to pyproject.toml using
uv add --dev pytest

--dev marks pytest as a development-only dependency, needed for testing but not for running the model.


uv run pytest                                   # everything under the current folder
uv run pytest tests/unit                        # just unit tests
uv run pytest tests/unit/test_gemma_norm.py     # one file
uv run pytest -k adaptive                       # only tests whose name contains "adaptive"
uv run pytest -q                                # quieter output; -v for one line per test

One small tip: add this to pyproject.toml so a uv run pytest only looks in tests/
[tool.pytest.ini_options]
testpaths = ["tests"]
"""

def test_plain_unit_rms_at_init():
    x = torch.randn(2, 5, 16) * 7
    out, gate = RMSNorm(16)(x)
    assert gate is None and out.shape == x.shape
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(2, 5), atol=1e-4)

def test_adaptive_broadcasts_cond_over_tokens():
    norm = RMSNorm(16, adaptive=True)
    torch.nn.init.normal_(norm.dense.weight)            # make it non-trivial
    out, gate = norm(torch.randn(2, 5, 16), cond=torch.randn(2, 16))
    assert out.shape == (2, 5, 16) and gate.shape == (2, 1, 16)

def test_keeps_input_dtype():
    out, _ = RMSNorm(16)(torch.randn(2, 16, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16

def test_wrong_cond_usage_raises():
    with pytest.raises(ValueError):
        RMSNorm(16)(torch.randn(2, 16), cond=torch.randn(2, 16))
    with pytest.raises(ValueError):
        RMSNorm(16, adaptive=True)(torch.randn(2, 16))