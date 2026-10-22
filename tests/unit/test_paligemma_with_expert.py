from dataclasses import replace

import pytest
import torch
from pi05.models.gemma import RMSNorm
from pi05.models.gemma_config import GemmaConfig
from pi05.models.paligemma_with_expert import PaliGemmaWithExpert
from pi05.models.siglip_config import SiglipConfig

VISION = SiglipConfig(image_size=28, patch_size=14, width=24, depth=2, mlp_dim=48, num_heads=4)
VLM = GemmaConfig(width=32, depth=2, mlp_dim=64, num_heads=4, num_kv_heads=1, head_dim=8, vocab_size=100)
EXPERT = GemmaConfig(width=16, depth=2, mlp_dim=32, num_heads=4, num_kv_heads=1, head_dim=8,
                     vocab_size=None, use_adarms=True)
B, LP, LS = 2, 5, 3      # batch, prefix tokens, suffix tokens
L = LP + LS

def make_model():
    model = PaliGemmaWithExpert(VISION, VLM, EXPERT)
    # adaRMS gates start at zero (expert = identity), which would make the suffix ignore
    # the prefix and hide cache bugs. Give the gates real values.
    for m in model.modules():
        if isinstance(m, RMSNorm) and m.adaptive:
            torch.nn.init.normal_(m.dense.weight, std=0.5)
    return model

def inputs():
    prefix = torch.randn(B, LP, VLM.width)
    suffix = torch.randn(B, LS, EXPERT.width)
    cond = torch.randn(B, EXPERT.width)
    positions = torch.arange(L).expand(B, L)
    mask = torch.ones(B, L, L, dtype=torch.bool)
    mask[:, :LP, LP:] = False                    # prefix may not see suffix (as in pi0.5)
    return prefix, suffix, positions, mask, cond

def test_inference_path_matches_training_path():
    model = make_model()
    prefix, suffix, positions, mask, cond = inputs()

    # training path
    prefix_joint, suffix_joint = model(prefix, suffix, positions, mask, cond)
    assert prefix_joint.shape == (B, LP, VLM.width) and suffix_joint.shape == (B, LS, EXPERT.width)

    # inference path: prefix once, then suffix twice with the same cache (two denoising steps)
    prefix_out, kv_cache = model.forward_prefix(prefix, positions[:, :LP], mask[:, :LP, :LP])
    step1 = model.forward_suffix(suffix, positions[:, LP:], mask[:, LP:, :], cond, kv_cache)
    step2 = model.forward_suffix(suffix, positions[:, LP:], mask[:, LP:, :], cond, kv_cache)

    assert torch.allclose(prefix_out, prefix_joint, atol=1e-5)
    assert torch.allclose(step1, suffix_joint, atol=1e-5)
    assert torch.equal(step1, step2)                                # cache not modified by a step


def test_gradient_checkpointing_gives_same_gradients():
    model = make_model().train()
    prefix, suffix, positions, mask, cond = inputs()
    grads = []
    for flag in (False, True):
        model.set_gradient_checkpointing(flag)
        model.zero_grad()
        _, suffix_out = model(prefix, suffix, positions, mask, cond)
        suffix_out.sum().backward()
        grads.append(model.expert.layers[0].mlp.up_proj.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_mismatched_configs_rejected():
    with pytest.raises(ValueError):
        PaliGemmaWithExpert(VISION, VLM, replace(EXPERT, depth=3))