"""
Tests for the policy server: Policy + B1KPolicy on CPU, and the websocket protocol.
"""

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import safetensors.torch
import torch
from websockets.sync.client import connect

from pi05.models.pi05 import Pi05
from pi05.serving.policy import CAMERA_KEYS, GRIPPER_DIMS, PROPRIO_KEY, B1KPolicy, Policy
from pi05.serving.websocket_server import WebsocketPolicyServer, packb, unpackb
from pi05.train_config import get_train_config

TOKENIZER_PATH = Path("pretrained/paligemma_tokenizer.model")
ACTION_DIMS = 23


def fake_obs(rng: np.random.Generator) -> dict:
    """
    A BEHAVIOR-style observation: 256-dim proprio and three 224x224 RGBA images.
    """
    obs = {PROPRIO_KEY: rng.standard_normal(256).astype(np.float32)}
    for key in CAMERA_KEYS:
        obs[key] = rng.integers(0, 256, size=(224, 224, 4), dtype=np.uint8)
    return obs


def write_tiny_checkpoint(checkpoint_dir: Path, model_config) -> None:
    """
    model.safetensors of a random tiny model, plus norm stats mapping [-1, 1] to itself.
    """
    torch.manual_seed(0)
    safetensors.torch.save_model(Pi05(model_config), str(checkpoint_dir / "model.safetensors"))
    stats = {
        "mean": [0.0] * ACTION_DIMS,
        "std": [1.0] * ACTION_DIMS,
        "q01": [-1.0] * ACTION_DIMS,
        "q99": [1.0] * ACTION_DIMS,
    }
    norm_stats = {"state": stats, "actions": stats, "num_frames": 1}
    (checkpoint_dir / "norm_stats.json").write_text(json.dumps(norm_stats))


def test_ensemble_averages_current_step_and_keeps_newest_grippers():
    """
    An older chunk contributes its index-1 action, the newest its index-0 action.
    """
    wrapper = B1KPolicy(policy=None, prompt="")
    older = np.zeros((50, ACTION_DIMS), dtype=np.float32)
    older[1] = 1.0  # made one step ago: its prediction for now is at index 1
    newer = np.zeros((50, ACTION_DIMS), dtype=np.float32)
    newer[0] = 3.0
    wrapper.chunks = deque([older, newer], maxlen=10)

    action = wrapper._ensemble_action()

    weights = np.exp(0.005 * np.arange(2))
    weights = weights / weights.sum()
    expected = weights[0] * 1.0 + weights[1] * 3.0
    for dim in range(ACTION_DIMS):
        if dim in GRIPPER_DIMS:
            assert action[dim] == pytest.approx(3.0)  # newest chunk only
        else:
            assert action[dim] == pytest.approx(expected)


@pytest.mark.skipif(not TOKENIZER_PATH.exists(), reason="needs the PaliGemma tokenizer (setup.md 2.1)")
def test_b1k_policy_acts_and_resets(tmp_path):
    """
    Tiny model on CPU: BEHAVIOR obs -> one (1, 23) action per call; reset clears chunks.
    """
    model_config = get_train_config("dev").model_config
    write_tiny_checkpoint(tmp_path, model_config)
    policy = Policy(tmp_path, model_config, TOKENIZER_PATH, torch.device("cpu"), dtype=torch.float32)
    wrapper = B1KPolicy(policy, prompt="turn on the radio")
    rng = np.random.default_rng(0)

    for _ in range(3):
        action = wrapper.act(fake_obs(rng))
        assert action.shape == (1, ACTION_DIMS)
        assert action.dtype == np.float32
        assert np.isfinite(action).all()
    assert len(wrapper.chunks) == 3
    assert wrapper.chunks[0].shape == (model_config.action_horizon, ACTION_DIMS)

    wrapper.reset()
    assert len(wrapper.chunks) == 0


class EchoPolicy:
    """
    Stand-in policy: returns the first proprio value as the action; counts resets.
    """

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, obs):
        return np.full((1, ACTION_DIMS), obs[PROPRIO_KEY][0], dtype=np.float32)


def free_port() -> int:
    """
    A port nobody is using right now (the OS picks one for port 0).
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_for_health(port: int, timeout_s: float = 10.0) -> int:
    """
    Poll /healthz like BEHAVIOR's client does, until the server answers.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as response:
                return response.status
        except (urllib.error.URLError, ConnectionError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.1)


def test_server_speaks_behavior_protocol():
    """
    Health check, metadata frame, reset with no reply, then obs -> {"action": ...}.
    """
    policy = EchoPolicy()
    port = free_port()
    server = WebsocketPolicyServer(policy, host="127.0.0.1", port=port)
    threading.Thread(target=server.serve_forever, daemon=True).start()  # dies with the test

    assert wait_for_health(port) == 200
    obs = fake_obs(np.random.default_rng(0))
    with connect(f"ws://127.0.0.1:{port}", compression=None, max_size=None) as conn:
        assert unpackb(conn.recv()) == {}  # metadata, sent once on connect
        conn.send(packb({"reset": True}))
        conn.send(packb(obs))
        reply = unpackb(conn.recv())  # if the server had replied to reset, this would be it

    assert policy.resets == 1
    np.testing.assert_array_equal(reply["action"], policy.act(obs))