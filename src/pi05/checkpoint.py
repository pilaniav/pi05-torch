"""
Training checkpoints: save, find latest, load.
"""

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path

import safetensors.torch
import torch
from safetensors import safe_open

from pi05.train_config import TrainConfig

logger = logging.getLogger(__name__)


# (openpi prefix, our prefix). Keys matching none are kept as-is (action/time heads).
OPENPI_RENAMES = [
    ("paligemma_with_expert.paligemma.model.vision_tower.",
     "paligemma_with_expert.paligemma.vision.model."),
    ("paligemma_with_expert.paligemma.model.multi_modal_projector.linear.",
     "paligemma_with_expert.paligemma.projector."),
    ("paligemma_with_expert.paligemma.model.language_model.",
     "paligemma_with_expert.paligemma.llm."),
    # Tied to embed_tokens; safetensors may store the shared tensor under either name.
    ("paligemma_with_expert.paligemma.lm_head.",
     "paligemma_with_expert.paligemma.llm.embed_tokens."),
    ("paligemma_with_expert.gemma_expert.model.",
     "paligemma_with_expert.expert."),
]
# Unused: the expert's lm_head is random in openpi (never converted, never used).
OPENPI_SKIP = ["paligemma_with_expert.gemma_expert.lm_head."]


def openpi_to_ours(key: str) -> str | None:
    """
    Our parameter name for an openpi key, or None if the key is unused.
    """
    for prefix in OPENPI_SKIP:
        if key.startswith(prefix):
            return None
    for openpi_prefix, our_prefix in OPENPI_RENAMES:
        if key.startswith(openpi_prefix):
            return our_prefix + key[len(openpi_prefix):]
    return key


def load_pretrained(model: torch.nn.Module, path: Path) -> None:
    """
    Load openpi's PyTorch pi0.5 weights (model.safetensors) into our model.

    Strict: every key must map to one of our params with the same shape, and every
    param must be loaded. Reads one tensor at a time to keep CPU memory low.
    """
    params = model.state_dict()  # name -> tensor sharing storage with the model
    loaded = set()

    with safe_open(str(path), framework="pt") as f:
        for openpi_key in f.keys():
            our_key = openpi_to_ours(openpi_key)
            if our_key is None:
                continue
            if our_key not in params:
                raise KeyError(f"{openpi_key} -> {our_key}: no such parameter in our model")
            tensor = f.get_tensor(openpi_key)
            if tensor.shape != params[our_key].shape:
                raise ValueError(
                    f"{our_key}: shape {tuple(tensor.shape)} != {tuple(params[our_key].shape)}"
                )
            with torch.no_grad():
                params[our_key].copy_(tensor)  # also casts dtype, e.g. bf16 -> fp32
            loaded.add(our_key)

    missing = sorted(set(params) - loaded)
    if missing:
        raise KeyError(f"{len(missing)} params not in checkpoint, e.g. {missing[:5]}")
    logger.info("Loaded pretrained weights from %s", path)


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    extra_files: tuple[Path, ...] = (),
) -> Path:
    """
    Save to checkpoint_dir/<step>/ atomically. Pass the unwrapped model, not DDP.

    extra_files (e.g. norm_stats.json) are copied in, so the checkpoint is self-contained.
    """
    final_dir = checkpoint_dir / str(step)
    tmp_dir = checkpoint_dir / f"tmp_{step}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    safetensors.torch.save_model(model, str(tmp_dir / "model.safetensors"))
    torch.save(optimizer.state_dict(), tmp_dir / "optimizer.pt")
    metadata = {"step": step, "config": asdict(config)}
    (tmp_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
    for path in extra_files:
        shutil.copyfile(path, tmp_dir / path.name)

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)

    logger.info("Saved checkpoint %s", final_dir)
    
    return final_dir

def saved_steps(checkpoint_dir: Path) -> list[int]:
    """
    All saved steps, oldest first. Ignores tmp_* folders from interrupted saves.
    """
    if not checkpoint_dir.exists():
        return []
    steps = []
    for path in checkpoint_dir.iterdir():
        if path.is_dir() and path.name.isdigit():
            steps.append(int(path.name))
    return sorted(steps)

def latest_step(checkpoint_dir: Path) -> int | None:
    """
    Highest saved step, or None if there is no checkpoint.
    """
    steps = saved_steps(checkpoint_dir)
    return steps[-1] if steps else None

def prune_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    """
    Delete all but the newest keep_last checkpoints.
    """
    if keep_last < 1:
        raise ValueError(f"keep_last must be >= 1, got {keep_last}")
    steps = saved_steps(checkpoint_dir)
    for step in steps[:-keep_last]:
        shutil.rmtree(checkpoint_dir / str(step))
        logger.info("Deleted old checkpoint %s", checkpoint_dir / str(step))

def load_checkpoint(
    checkpoint_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> None:
    """
    Load model and optimizer state in place from checkpoint_dir/<step>/.
    """
    step_dir = checkpoint_dir / str(step)
    safetensors.torch.load_model(model, str(step_dir / "model.safetensors"), device=str(device))
    optimizer_state = torch.load(step_dir / "optimizer.pt", map_location=device)
    optimizer.load_state_dict(optimizer_state)
    logger.info("Loaded checkpoint %s", step_dir)