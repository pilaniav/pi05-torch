"""
Training dataset: one sample = one frame of one episode, ready for the model.

How dataset[index] works:
1. Find the sample: index is a global frame number over all episodes of the split.
   The episode is the one whose frame range contains it; t = index - its first frame.
2. State -> prompt: the episode's 23-dim state at t (extracted and normalized once in
   __init__) is discretized into the prompt together with the task text, and tokenized.
3. Actions -> targets: the normalized actions at t .. t+49, clamped to the episode's last
   frame (repeated steps marked False in action_mask), padded from 23 to 32 dims.
4. Images: frame t of each camera's prepared 224 video, scaled to [-1, 1]; augmented when
   augment=True (crop + rotation for the head camera only, as openpi).
5. Returned as a dict of tensors; the loader's collate stacks a batch of them.
"""


import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchcodec.decoders import VideoDecoder

from pi05.data_pipeline.b1k_state_action import extract_state, read_state_actions
from pi05.data_pipeline.download import read_jsonl
from pi05.data_pipeline.norm_stats import NORM_STATS_FILE
from pi05.data_pipeline.prepare_dataset import episode_paths
from pi05.data_pipeline.transforms import (
    PromptTokenizer,
    load_norm_stats,
    normalize,
    pad_last_dim,
    prepare_image,
)
from pi05.distributed import DistInfo
from pi05.models.observation import Observation


CAMERAS = ("head", "left_wrist", "right_wrist")


def find_episode_and_frame(starts: np.ndarray, index: int) -> tuple[int, int]:
    """
    Find which episode a global frame index belongs to, and where inside it.

    Args:
        starts: (num_episodes,) global index of each episode's first frame, in
                episode order, e.g. [0, 1956, 4166] for episode lengths 1956, 2210, ...
        index: global frame index, e.g. sampled by the sampler. 0 <= index < total frames.

    Returns:
        (row, t): row is the episode's position in the episode list (0, 1, 2, ...),
        t is the frame number inside that episode (0 = its first frame).

    Example:
        starts = [0, 1956, 4166]
        index 1955 -> (0, 1955)   last frame of the first episode
        index 1956 -> (1, 0)      first frame of the second episode
        index 4200 -> (2, 34)
    """
    row = int(np.searchsorted(starts, index, side="right")) - 1
    return row, index - int(starts[row])


class B1kDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        tokenizer_path: Path,
        augment: bool,
        action_horizon: int = 50,
        action_dim: int = 32,
        max_token_len: int = 200
    ):
        self.dataset_dir = dataset_dir
        self.info = json.loads((dataset_dir / "meta/info.json").read_text())
        self.augment = augment
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.tokenizer = PromptTokenizer(tokenizer_path, max_token_len)
        norm_stats = load_norm_stats(dataset_dir / NORM_STATS_FILE)

        self.episode_indices = []
        self.tasks = []
        lengths = []
        states = []
        actions = []
        for record in read_jsonl(dataset_dir / "meta/episodes.jsonl"):
            if record["split"] != split:
                continue
            parquet, _ = episode_paths(dataset_dir, self.info, record["episode_index"], CAMERAS)
            raw_state, episode_actions = read_state_actions(parquet)
            states.append(extract_state(raw_state))
            actions.append(episode_actions)
            self.episode_indices.append(record["episode_index"])
            self.tasks.append(record["tasks"][0])
            lengths.append(record["length"])
        if not lengths:
            raise ValueError(f"No {split!r} episodes in {dataset_dir}")

        self.lengths = np.array(lengths)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        # All frames of the split back to back, normalized once: (num_frames, 23) each.
        self.states = normalize(np.concatenate(states), norm_stats["state"])
        self.actions = normalize(np.concatenate(actions), norm_stats["actions"])
    
    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row, t = find_episode_and_frame(self.starts, index)
        start = int(self.starts[row])
        length = int(self.lengths[row])

        # State -> prompt. states holds all frames back to back, so index is its row.
        prompt_ids, prompt_mask = self.tokenizer.tokenize(self.tasks[row], self.states[index])

        # Actions -> targets: t .. t+49, clamped to the episode's last frame.
        steps = np.arange(t, t + self.action_horizon)
        action_mask = steps < length  # False for steps past the episode end (repeats)
        frames = start + np.minimum(steps, length - 1)  # global frame of each step
        actions = pad_last_dim(self.actions[frames], self.action_dim)  # (50, 23) -> (50, 32)

        # Images: frame t of each camera's prepared video.
        _, video_paths = episode_paths(
            self.dataset_dir, self.info, self.episode_indices[row], CAMERAS
        )
        images = []
        for camera in CAMERAS:
            frame = VideoDecoder(str(video_paths[camera]))[t]  # uint8 (3, 224, 224)
            images.append(
                prepare_image(frame, train=self.augment, crop_and_rotate=(camera == "head"))
            )

        # One sample as tensors; collate stacks a batch of them.
        return {
            "images": torch.stack(images),  # (3 cameras, 3, 224, 224) float32 in [-1, 1]
            "image_masks": torch.ones(len(CAMERAS), dtype=torch.bool),
            "prompt_ids": torch.from_numpy(prompt_ids),  # (200,) int64
            "prompt_mask": torch.from_numpy(prompt_mask),  # (200,) bool
            "actions": torch.from_numpy(actions),  # (50, 32) float32, normalized
            "action_mask": torch.from_numpy(action_mask),  # (50,) bool
        }


def batch_to_observation(batch: dict[str, torch.Tensor]) -> Observation:
    return Observation(
        images=batch["images"],
        image_masks=batch["image_masks"],
        prompt_ids=batch["prompt_ids"],
        prompt_mask=batch["prompt_mask"],
    )


def collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Stack a list of samples into one batch: each value gets a leading batch dimension.
    """
    batch = {}
    for key in samples[0]:
        values = []
        for sample in samples:
            values.append(sample[key])
        batch[key] = torch.stack(values)
    return batch


def make_data_loader(
    dataset: B1kDataset,
    batch_size: int,
    info: DistInfo,
    num_workers: int,
    seed: int,
) -> Iterator[tuple[Observation, torch.Tensor, torch.Tensor]]:
    """
    Endless iterator of (observation, actions, action_mask) batches for this rank.

    batch_size is per rank. Each rank sees a different, equally sized share of the
    data, reshuffled every epoch.
    """
    sampler = DistributedSampler(
        dataset,
        num_replicas=info.world_size,
        rank=info.rank,
        shuffle=True,
        seed=seed,
        drop_last=True,  # same number of samples on every rank: no rank runs out first
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=info.device.type == "cuda",  # faster copies to the GPU
        drop_last=True,  # no smaller last batch
        persistent_workers=num_workers > 0,  # keep workers alive between epochs
        multiprocessing_context="spawn" if num_workers > 0 else None  # fresh processes, no fork
    )

    epoch = 0
    while True:
        sampler.set_epoch(epoch)  # a different shuffle each epoch, same on all ranks
        for batch in loader:
            observation = batch_to_observation(batch)
            yield observation, batch["actions"], batch["action_mask"]
        epoch += 1


def make_val_batches(
    dataset: B1kDataset, num_batches: int, batch_size: int, seed: int
) -> list[tuple[Observation, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Fixed validation batches: (observation, actions, noise, t), identical at every evaluation.

    The validation split has 10 episodes per task (20 for two tasks, ~40,000 frames):
    too many to evaluate often. So we pick num_batches * batch_size frames, evenly spread
    over the split (all episodes covered). Noise and t are drawn once from seed, so values
    from different steps are comparable. Use a dataset with augment=False.

    Example: num_batches=10, batch_size=4 -> 40 frames, ~1,000 apart:
        indices = [0, 1025, 2051, 3077, 4102, 5128, 6153, 7179, ...]

        outer pass 1: start = 0
            indices[0:4] = [0, 1025, 2051, 3077]       inner loop reads these 4 frames
            collate -> batch 1
        outer pass 2: start = 4
            indices[4:8] = [4102, 5128, 6153, 7179]    inner loop reads these 4 frames
            collate -> batch 2
        ... up to batch 10
    """
    indices = np.linspace(0, len(dataset) - 1, num_batches * batch_size).astype(int)
    generator = torch.Generator().manual_seed(seed)

    batches = []
    for start in range(0, len(indices), batch_size):  # one pass per batch
        samples = []
        for index in indices[start : start + batch_size]:  # each frame of this batch
            samples.append(dataset[int(index)])
        batch = collate(samples)
        noise = torch.randn(batch["actions"].shape, generator=generator)
        t = torch.rand(len(samples), generator=generator) * 0.999 + 0.001  # in (0, 1]
        batches.append((batch_to_observation(batch), batch["actions"], noise, t))
    return batches