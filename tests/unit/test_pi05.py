import torch
from pi05.models.gemma import RMSNorm
from pi05.models.observation import Observation
from pi05.models.pi05 import Pi05, sinusoidal_time_embedding
from pi05.models.pi05 import make_attention_mask, make_positions
from pi05.models.pi05 import sample_time
from pi05.models.pi05_config import Pi05Config

TINY = Pi05Config(
    vision_variant="siglip_tiny",
    vlm_variant="gemma_tiny",
    expert_variant="gemma_tiny_expert",
    action_dim=6,
    action_horizon=4,
)

def make_obs(B=2):
    return Observation(
        images=torch.randn(B, 3, 3, 28, 28),
        image_masks=torch.ones(B, 3, dtype=torch.bool),
        prompt_ids=torch.randint(0, 100, (B, 5)),
        prompt_mask=torch.ones(B, 5, dtype=torch.bool),
    )

def randomize_adarms_gates(model):
    """
    Zero-init adaRMS gates make the expert ignore the prefix and hide bugs; give them values.
    """
    for m in model.modules():
        if isinstance(m, RMSNorm) and m.adaptive:
            torch.nn.init.normal_(m.dense.weight, std=0.5)

def test_time_embedding_shape_and_t0():
    emb = sinusoidal_time_embedding(
        torch.Tensor([0.0, 0.3, 1.0]), dim=16, min_period=4e-3, max_period=4.0
    )
    assert emb.shape == (3, 16)
    assert torch.allclose(emb[0, :8], torch.zeros(8))  # t = 0: sin half is 0
    assert torch.allclose(emb[0, 8:], torch.ones(8))  # t = 0: cos half is 1


def test_embed_suffix_shapes():
    model = Pi05(TINY)
    embs, cond = model.embed_suffix(torch.randn(2, 4, 6), torch.rand(2))
    assert embs.shape == (2, 4, 16) and cond.shape == (2, 16)  # 16 = tiny expert width

def test_attention_mask_pi05_pattern():
    # prefix: 2 real + 1 padding; suffix: 2 actions
    pad = torch.tensor([[True, True, False, True, True]])
    block_start = torch.tensor([[False, False, False, True, False]])
    m = make_attention_mask(pad, block_start)[0]  # (5, 5): [query, key]
    assert m[0, 1] and m[1, 0]  # prefix tokens see each other
    assert not m[0, 3]          # prefix can't see suffix
    assert m[3, 0] and m[4, 3] and m[3, 4]  # suffix sees prefix, and both actions see each other
    assert not m[:, 2].any() and not m[2].any()  # padding: never seen, and sees nothing

def test_positions_skip_padding():
    pad = torch.tensor([[True, True, False, True]])
    assert make_positions(pad).tolist() == [[0, 1, 1, 2]]

def test_embed_prefix_shapes_and_pad():
    model = Pi05(TINY)   # tiny SigLIP: 28x28 images -> 4 patches
    obs = Observation(
        images=torch.randn(2, 3, 3, 28, 28),
        image_masks=torch.tensor([[True, False, True], [True, True, True]]), # sample 0: camera 1 missing
        prompt_ids=torch.randint(0, 100, (2, 5)),
        prompt_mask=torch.tensor([[True] * 3 + [False] * 2] * 2),
    )
    embs, pad = model.embed_prefix(obs)
    assert embs.shape == (2, 3 * 4 + 5, 32)  # 3 cameras x 4 patches + 5 prompt, VLM width 32
    assert not pad[0, 4:8].any() and pad[0, :4].all()  # sample 0, camera 1's patches are padding
    assert pad[1, :12].all()  # sample 1: all cameras present
    assert pad[:, 12:].tolist() == obs.prompt_mask.tolist()

def test_sample_time_range():
    t = sample_time(1000, torch.device("cpu"))
    assert t.shape == (1000,) and t.min() >= 0.001 and t.max() <= 1.0

def test_forward_loss_shape_and_determination():
    model = Pi05(TINY)
    obs, actions = make_obs(), torch.randn(2, 4, 6)
    noise, t = torch.randn(2, 4, 6), torch.rand(2)
    loss1 = model(obs, actions, noise=noise, t=t)
    loss2 = model(obs, actions, noise=noise, t=t)
    assert loss1.shape == (2, 4, 6) and loss1.dtype == torch.float32
    assert torch.equal(loss1, loss2)  # fixed noise/t -> same loss

def test_gradients_reach_all_parts():
    model = Pi05(TINY)
    randomize_adarms_gates(model)
    model(make_obs(), torch.randn(2, 4, 6)).mean().backward()
    pg = model.paligemma_with_expert.paligemma
    for p in (model.action_in_proj.weight, model.time_mlp_in.weight, pg.projector.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0

def test_sample_actions_matches_training_path():
    model = Pi05(TINY)
    randomize_adarms_gates(model)
    obs, noise = make_obs(), torch.randn(2, 4, 6)

    actions = model.sample_actions(obs, noise=noise, num_steps=1)  # one step: a = noise - v
    assert actions.shape == (2, 4, 6)

    loss = model(obs, actions, noise=noise, t=torch.ones(2))  # x_t = noise, target = noise - a = v
    assert loss.max() < 1e-8                                  # inference and training agree