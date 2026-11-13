"""
Training: LR schedule, optimizer, train step, loop, entry point.
"""
import itertools
import logging
import math
import random
import time
import tyro
from collections.abc import Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pi05.checkpoint import (
    latest_step,
    load_checkpoint,
    load_pretrained,
    prune_checkpoints,
    save_checkpoint,
)
from pi05.data_pipeline.dataset import B1kDataset, make_data_loader, make_val_batches
from pi05.distributed import DistInfo, cleanup_distributed, setup_distributed
from pi05.fake_data import fake_batches
from pi05.log_setup import init_logging
from pi05.models.observation import Observation
from pi05.models.pi05 import Pi05
from pi05.train_config import TRAIN_CONFIGS, TrainConfig
from pi05.data_pipeline.norm_stats import NORM_STATS_FILE


DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}

logger = logging.getLogger("pi05.train") 


def lr_at(step: int, config: TrainConfig) -> float:
    """
    Linear warmup then cosine decay (same as openpi / optax warmup_cosine_decay).
    """
    peak = config.peak_lr
    warmup = config.warmup_steps
    decay_steps = config.decay_steps

    if step < warmup:
        start = peak / (warmup + 1)
        return start + (peak - start) * step / warmup

    progress = (step - warmup) / max(1, decay_steps - warmup)
    progress = min(progress, 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return config.end_lr + (peak - config.end_lr) * cosine


def freeze_unused_params(model: Pi05) -> None:
    """
    Freeze params that never affect the loss.

    In the VLM's last layer, the prefix tokens' queries, attention output, MLP and the 
    final norm only produce prefix_out, which pi0.5 discards. They would get no gradient,
    and DDP requires a gradient for every trainable param. The last layer's K/V are
    still used (the suffix attends to them).
    """
    llm = model.paligemma_with_expert.paligemma.llm
    last_layer = llm.layers[-1]
    unused = [
        last_layer.self_attn.q_proj,
        last_layer.self_attn.o_proj,
        last_layer.post_attention_layernorm,
        last_layer.mlp,
        llm.norm,
    ]
    for module in unused:
        module.requires_grad_(False)


def make_optimizer(model: Pi05, config: TrainConfig) -> torch.optim.AdamW:
    """
    AdamW over trainable params. LR is overwritten every step by lr_at.
    """
    params = []
    for param in model.parameters():
        if param.requires_grad:
            params.append(param)

    return torch.optim.AdamW(
        params,
        lr=config.peak_lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def train_step(
        model: torch.nn.Module,
        obs: Observation,
        actions: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        step: int,
        config: TrainConfig
) -> dict[str, torch.Tensor]:
    """
    One optimization step. Inputs must already be on the model's device.
    """
    lr = lr_at(step, config)
    for group in optimizer.param_groups:
        group["lr"] = lr

    dtype = DTYPES[config.precision]
    enabled = dtype != torch.float32
    # Mixed precision: autocast runs matmuls in bf16 (faster, less memory) and
    # precision-sensitive ops (softmax, norms, loss) in fp32. The weights stay fp32,
    # so small updates aren't rounded away. Disabled when precision is "float32".
    with torch.autocast(device_type=actions.device.type, dtype=dtype, enabled=enabled):
        loss = model(obs, actions).mean()

    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    return {"loss": loss.detach(), "grad_norm": grad_norm, "lr": torch.tensor(lr)}


@torch.no_grad()
def validation_loss(
    model: Pi05,
    val_batches: list[tuple[Observation, torch.Tensor, torch.Tensor, torch.Tensor]],
    device: torch.device,
    config: TrainConfig,
) -> float:
    """
    Mean flow-matching loss on the fixed validation batches (unwrapped model, no DDP).
    """
    dtype = DTYPES[config.precision]
    model.eval()
    total = 0.0
    for obs, actions, noise, t in val_batches:
        # Same mixed precision as training: autocast runs matmuls in bf16 and
        # precision-sensitive ops in fp32. Disabled when precision is "float32".
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype != torch.float32):
            losses = model(
                obs.to(device), actions.to(device), noise=noise.to(device), t=t.to(device)
            )   # (B, H, D): squared error, predicted vs target velocity
        total += losses.mean().item()
    model.train()
    return total / len(val_batches)


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy and torch (CPU and all GPUs).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds every CUDA device

def train_loop(
    config: TrainConfig,
    data: Iterator[tuple[Observation, torch.Tensor, torch.Tensor]],
    info: DistInfo,
    val_batches: list | None = None,  # rank 0 only; None = no validation
) -> Pi05:
    """
    Train for config.num_train_steps. data yields per-rank (obs, actions, action_mask) batches.
    """
    set_seed(config.seed + info.rank)
    model = Pi05(config.model_config)

    if config.pretrained_path is not None:
        load_pretrained(model, config.pretrained_path)
    
    freeze_unused_params(model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Parameters: %.1fM trainable / %.1fM total", trainable / 1e6, total / 1e6)

    model.set_gradient_checkpointing(config.gradient_checkpointing)
    model.to(info.device)
    model.train()
    optimizer = make_optimizer(model, config)

    start_step = 0
    saved_step = latest_step(config.checkpoint_dir)
    if saved_step is not None:
        if not config.resume:
            raise FileExistsError(
                f"{config.checkpoint_dir} has checkpoints: pass --resume, or use a new --exp-name"
            )
        load_checkpoint(config.checkpoint_dir, saved_step, model, optimizer, info.device)
        start_step = saved_step

    ddp_model = model
    if info.world_size > 1:
        device_ids = [info.local_rank] if info.device.type == "cuda" else None
        ddp_model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            gradient_as_bucket_view=True,  # grads live in DDP's buffers: saves one full copy (~13 GB)
        )

    if config.compile:
        ddp_model = torch.compile(ddp_model)

    checkpoint_extras = ()
    if config.dataset_dir is not None:
        checkpoint_extras = (config.dataset_dir / NORM_STATS_FILE,)  # needed at inference

    loss_sum = torch.tensor(0.0, device=info.device)
    interval_start = time.time()

    for step in range(start_step, config.num_train_steps):
        # action_mask is not used: like openpi, steps past an episode's end are trained on
        # as "hold the final pose" (the last action repeated).
        obs, actions, _action_mask = next(data)
        obs, actions = obs.to(info.device), actions.to(info.device)
        metrics = train_step(ddp_model, obs, actions, optimizer, step, config)
        loss_sum += metrics["loss"]
        done = step + 1

        if done % config.log_interval == 0:
            if info.world_size > 1:
                # Collective: every rank must call it; early ranks wait for the others.
                dist.all_reduce(loss_sum)  # sum over ranks
            
            if info.is_main:
                gpu_mem_gb = 0.0  # stays 0 on CPU
                if info.device.type == "cuda":
                    gpu_mem_gb = torch.cuda.max_memory_allocated(info.device) / 1e9
                logger.info(
                    "step %d  loss %.4f  grad_norm %.3f  lr %.2e  %.2fs/step  mem %.1fGB",
                    done,
                    loss_sum.item() / (config.log_interval * info.world_size),
                    metrics["grad_norm"].item(),
                    metrics["lr"].item(),
                    (time.time() - interval_start) / config.log_interval,
                    gpu_mem_gb,
                )

            loss_sum.zero_()
            interval_start = time.time()

        if val_batches and config.val_interval > 0 and done % config.val_interval == 0:
            val_loss = validation_loss(model, val_batches, info.device, config)
            logger.info("step %d  val_loss %.4f", done, val_loss)

            # TODO (research stage): keep the best checkpoint by val_loss (weights only, e.g.
            # checkpoint_dir/best/). Needed if val_loss rises late in a run (overfitting):
            # pruning keeps only the last keep_last_checkpoints, so the best one may be gone.
            # Until then, val_loss is in train.log; raise keep_last_checkpoints if needed.

        if info.is_main and (done % config.save_interval == 0 or done == config.num_train_steps):
            save_checkpoint(
                config.checkpoint_dir, done, model, optimizer, config, checkpoint_extras
            )
            prune_checkpoints(config.checkpoint_dir, config.keep_last_checkpoints)

    if info.world_size > 1:
        dist.barrier()  # wait for rank 0 to finish the final save

    return model


def main() -> None:
    choices = {}
    for name, config in TRAIN_CONFIGS.items():
        choices[name] = (name, config)
    config = tyro.extras.overridable_config_cli(choices)

    info = setup_distributed()

    init_logging(info, log_file=config.checkpoint_dir / "train.log")
    logger.info("World size %d, device %s", info.world_size, info.device)
    logger.info("Config: %s", config)

    if config.batch_size % info.world_size != 0:
        raise ValueError(
            f"batch_size {config.batch_size} not divisible by {info.world_size} ranks"
        )
    batch_size = config.batch_size // info.world_size

    val_batches = None
    if config.dataset_dir is None:
        data = fake_batches(config.model_config, batch_size, seed=config.seed + info.rank)
    else:
        model_config = config.model_config
        dataset = B1kDataset(
            config.dataset_dir,
            split="train",
            tokenizer_path=config.tokenizer_path,
            augment=True,
            action_horizon=model_config.action_horizon,
            action_dim=model_config.action_dim,
            max_token_len=model_config.max_token_len,
        )
        data = make_data_loader(
            dataset, batch_size, info, config.num_workers, seed=config.seed
        )

        if info.is_main and config.val_interval > 0:
            val_dataset = B1kDataset(
                config.dataset_dir,
                split="val",
                tokenizer_path=config.tokenizer_path,
                augment=False,
                action_horizon=model_config.action_horizon,
                action_dim=model_config.action_dim,
                max_token_len=model_config.max_token_len,
            )
            val_batches = make_val_batches(
                val_dataset, config.num_val_batches, batch_size, seed=config.seed
            )

    if config.overfit_one_batch:
        one_batch = next(data)
        data.close()  # stop the loader's workers: only this one batch is needed
        data = itertools.repeat(one_batch)  # the same batch forever
    
    try:
        train_loop(config, data, info, val_batches)
    finally:
        if hasattr(data, "close"):
            data.close()  # stop data loader workers before the process exits
        cleanup_distributed()  # runs even if training raises


if __name__ == "__main__":
    main()