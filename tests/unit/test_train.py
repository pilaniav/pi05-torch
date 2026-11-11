import os
import pytest
import socket
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

from helpers import TINY, make_obs, randomize_adarms_gates
from pi05.checkpoint import latest_step, load_checkpoint, save_checkpoint
from pi05.checkpoint import prune_checkpoints, saved_steps
from pi05.distributed import cleanup_distributed, setup_distributed
from pi05.fake_data import fake_batches
from pi05.models.pi05 import Pi05
from pi05.train import (
    freeze_unused_params,
    lr_at,
    make_optimizer,
    train_loop,
    train_step,
    validation_loss,
)
from pi05.train_config import TrainConfig


CONFIG = TrainConfig(
    exp_name="test", peak_lr=1.0, end_lr=0.1, warmup_steps=10, decay_steps=110
)

def free_port() -> int:
    """
    Ask the OS for an unused port for the processes to meet on.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]

def _ddp_worker(rank, world_size, port, checkpoint_base_dir):
    """
    Runs in each spawned process: set torchrun's env vars, train, compare weights.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    info = setup_distributed()
    try:
        config = TrainConfig(
            exp_name="ddp", model_config=TINY, checkpoint_base_dir=checkpoint_base_dir,
            num_train_steps=3, log_interval=1, peak_lr=1e-3, warmup_steps=0,
            decay_steps=100, precision="float32",
        )
        data = fake_batches(TINY, batch_size=2, seed=rank)  # different data per rank
        model = train_loop(config, data, info)

        params = []
        for param in model.parameters():
            params.append(param.detach().flatten())
        flat = torch.cat(params)

        gathered = [torch.empty_like(flat) for _ in range(world_size)]
        dist.all_gather(gathered, flat)  # every rank receives every rank's weights
        assert torch.equal(gathered[0], gathered[1])
    finally:
        cleanup_distributed()

@pytest.mark.parametrize(
    "step, expected",
    [
        (0, 1.0 / 11),  # warmup starts small, not at zero
        (10, 1.0),      # peak at end of warmup
        (60, 0.55),     # halfway through cosine: (peak + end) / 2
        (110, 0.1),     # end of decay
        (500, 0.1),     # stays at end_lr afterwards
    ]
)
def test_lr_schedule_key_points(step, expected):
    assert lr_at(step, CONFIG) == pytest.approx(expected)

def test_frozen_params_are_exactly_the_unused_ones():
    torch.manual_seed(0)
    model = Pi05(TINY)
    randomize_adarms_gates(model)  # zero gates at init would zero all VLM grads
    actions = torch.randn(2, TINY.action_horizon, TINY.action_dim)
    model(make_obs(), actions).mean().backward()

    unused = set()
    for name, param in model.named_parameters():
        if param.grad is None or torch.all(param.grad == 0):
            unused.add(name)

    freeze_unused_params(model)
    frozen = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen.add(name)

    assert frozen == unused

def test_train_step_overfits_one_batch():
    torch.manual_seed(0)
    config = TrainConfig(
        exp_name="test", model_config=TINY, peak_lr=1e-3, warmup_steps=0,
        decay_steps=100, precision="float32",
    )
    model = Pi05(TINY)
    freeze_unused_params(model)
    optimizer = make_optimizer(model, config)
    obs = make_obs()
    actions = torch.randn(2, TINY.action_horizon, TINY.action_dim)
    noise = torch.randn_like(actions)
    t = torch.full((2,), 0.5)

    def eval_loss():
        with torch.no_grad():
            return model(obs, actions, noise=noise, t=t).mean().item()

    before = eval_loss()
    for step in range(30):
        train_step(model, obs, actions, optimizer, step, config)
    after = eval_loss()

    assert after < 0.9 * before

def test_resume_continues_identically(tmp_path):
    config = TrainConfig(
        exp_name="test", model_config=TINY, checkpoint_base_dir=tmp_path,
        peak_lr=1e-3, warmup_steps=0, decay_steps=100, precision="float32",
    )
    obs = make_obs()
    actions = torch.randn(2, TINY.action_horizon, TINY.action_dim)

    def build():
        model = Pi05(TINY)
        freeze_unused_params(model)
        return model, make_optimizer(model, config)

    torch.manual_seed(0)
    model_a, optimizer_a = build()
    for step in range(2):
        train_step(model_a, obs, actions, optimizer_a, step, config)
    save_checkpoint(config.checkpoint_dir, 2, model_a, optimizer_a, config)

    torch.manual_seed(1)  # different init: loading must overwrite it
    model_b, optimizer_b = build()
    step = latest_step(config.checkpoint_dir)
    load_checkpoint(config.checkpoint_dir, step, model_b, optimizer_b, torch.device("cpu"))

    for model, optimizer in [(model_a, optimizer_a), (model_b, optimizer_b)]:
        torch.manual_seed(2)  # same noise and t for both
        train_step(model, obs, actions, optimizer, step, config)

    for param_a, param_b in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(param_a, param_b)

def test_prune_keeps_newest(tmp_path):
    for step in [1000, 3000, 2000]:  # unsorted on purpose
        (tmp_path / str(step)).mkdir()
    (tmp_path / "tmp_4000").mkdir()  # interrupted save: not a checkpoint

    prune_checkpoints(tmp_path, keep_last=2)

    assert saved_steps(tmp_path) == [2000, 3000]

def test_ddp_ranks_stay_in_sync(tmp_path):
    mp.spawn(_ddp_worker, args=(2, free_port(), tmp_path), nprocs=2)

def test_validation_loss_is_repeatable_and_leaves_model_unchanged():
    torch.manual_seed(0)
    config = TrainConfig(exp_name="test", model_config=TINY, precision="float32")
    model = Pi05(TINY)
    actions = torch.randn(2, TINY.action_horizon, TINY.action_dim)
    val_batches = [(make_obs(), actions, torch.randn_like(actions), torch.full((2,), 0.5))]
    before = [p.clone() for p in model.parameters()]

    first = validation_loss(model, val_batches, torch.device("cpu"), config)
    second = validation_loss(model, val_batches, torch.device("cpu"), config)

    assert first == second
    for old, new in zip(before, model.parameters()):
        assert torch.equal(old, new)
    assert model.training  # back in training mode afterwards