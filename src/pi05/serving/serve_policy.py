"""
Serve a trained pi0.5 checkpoint to the BEHAVIOR eval over a websocket.

One server answers for one task: the eval sends no prompt, so the task's prompt is
fixed at startup.

Usage:
    uv run python -m pi05.serving.serve_policy \
        --checkpoint-dir train_results/pi05_b1k_4xa100/50000 \
        --tasks-file train_results/pi05_b1k_4xa100/data/meta/tasks.jsonl \
        --task-name turning_on_radio
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from pi05.data_pipeline.download import read_jsonl
from pi05.serving.policy import B1KPolicy, Policy
from pi05.serving.websocket_server import WebsocketPolicyServer
from pi05.train_config import get_train_config

logger = logging.getLogger("pi05.serve")


@dataclass
class ServeConfig:
    checkpoint_dir: Path  # model.safetensors and the training run's norm_stats.json
    task_name: str  # the BEHAVIOR task this server answers for, e.g. turning_on_radio
    tasks_file: Path = Path("prepared_data/behavior/meta/tasks.jsonl")  # name -> prompt text
    config_name: str = "pi05_b1k"  # training config: model sizes and tokenizer path
    host: str = "0.0.0.0"
    port: int = 8000
    mode: Literal["temporal_ensemble", "receding_horizon"] = "temporal_ensemble"
    ensemble_size: int = 10  # temporal_ensemble: chunks averaged
    replan_steps: int = 10  # receding_horizon: actions executed per model call
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir: Path = Path("eval_results")  # eval outputs live here, like train_results
    save_obs: bool = False  # save the first observation to results_dir/<task>_obs.npz


def find_task_prompt(tasks_file: Path, task_name: str) -> str:
    """
    The exact prompt text the model was trained with for task_name.
    """
    for record in read_jsonl(tasks_file):
        if record["task_name"] == task_name:
            return record["task"]
    raise ValueError(f"task {task_name!r} not found in {tasks_file}")


def main(config: ServeConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    train_config = get_train_config(config.config_name)
    prompt = find_task_prompt(config.tasks_file, config.task_name)
    logger.info("Task %s, prompt: %r", config.task_name, prompt)

    device = torch.device(config.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    policy = Policy(
        config.checkpoint_dir,
        train_config.model_config,
        train_config.tokenizer_path,
        device,
        dtype,
    )
    logger.info("Loaded %s on %s (%s)", config.checkpoint_dir, device, dtype)

    wrapper = B1KPolicy(
        policy,
        prompt,
        mode=config.mode,
        ensemble_size=config.ensemble_size,
        replan_steps=config.replan_steps,
    )

    save_obs_path = None
    if config.save_obs:
        save_obs_path = config.results_dir / f"{config.task_name}_obs.npz"
    server = WebsocketPolicyServer(
        wrapper, host=config.host, port=config.port, save_obs_path=save_obs_path
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # Ctrl-C: normal way to stop the server
        logger.info("Server stopped")


if __name__ == "__main__":
    main(tyro.cli(ServeConfig))