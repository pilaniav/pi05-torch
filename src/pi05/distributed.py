"""
Distributed setup: one process per GPU, launched by torchrun.

torchrun --nproc_per_node=8 scripts/train.py ...             # 1 machine, 8 GPUs
torchrun --nnodes=2 --nproc_per_node=8 ... scripts/train.py  # 2 machines (+ address of the main one)
"""

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


# Long enough for rank 0 to save a ~40 GB checkpoint while the others wait.
DIST_TIMEOUT = timedelta(minutes=60)


@dataclass(frozen=True)
class DistInfo:
    rank: int             # global process index: 0 .. world_size - 1
    local_rank: int       # index on this machine = GPU index
    world_size: int       # total number of processes
    device: torch.device

    @property
    def is_main(self) -> bool:
        """
        Rank 0 does logging and checkpointing.
        """
        return self.rank == 0


def setup_distributed() -> DistInfo:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if world_size > 1:
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, timeout=DIST_TIMEOUT)

    return DistInfo(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()