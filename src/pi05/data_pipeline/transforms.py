"""
Per-sample transforms, shared by training (data loader) and inference (policy server).

Matches the data transforms openpi uses for its "pi05_b1k" training config
(openpi/src/openpi/training/config.py). Each function states its exact formula.
"""

import json
import logging
import math
import numpy as np
import sentencepiece
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


logger = logging.getLogger(__name__)

STATE_BINS = 256
QUANTILE_EPS = 1e-6  # openpi: avoids division by zero when q99 == q01


def load_norm_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """
    norm_stats.json -> {"state": {"q01": array, ...}, "actions": {...}} as float32 arrays.
    """
    raw = json.loads(path.read_text())
    stats = {}
    for key in ("state", "actions"):
        stats[key] = {}
        for name, values in raw[key].items():
            stats[key][name] = np.asarray(values, dtype=np.float32)
    return stats


def normalize(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """
    Quantile normalization: q01 -> -1, q99 -> +1. Not clipped (as openpi).
    """
    return (x - stats["q01"]) / (stats["q99"] - stats["q01"] + QUANTILE_EPS) * 2.0 - 1.0


def unnormalize(x: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    """
    Inverse of normalize (inference: model output -> robot actions).
    """
    return (x + 1.0) / 2.0 * (stats["q99"] - stats["q01"] + QUANTILE_EPS) + stats["q01"]


def pad_last_dim(x: np.ndarray, size: int) -> np.ndarray:
    """
    Zero-pad the last dimension to size, e.g. actions (50, 23) -> (50, 32).
    """
    if x.shape[-1] > size:
        raise ValueError(f"last dim {x.shape[-1]} is larger than {size}")
    padding = np.zeros(x.shape[:-1] + (size - x.shape[-1],), dtype=x.dtype) # (50,) + (9,) = (50, 9)
    return np.concatenate([x, padding], axis=-1)


class PromptTokenizer:
    """
    pi0.5 prompt: task text + discretized state -> token ids + mask, exactly as openpi.

    Each DataLoader worker gets its own copy of this object. The tokenizer file is
    opened the first time a worker uses it, so every worker loads its own.
    """

    def __init__(self, model_path: Path, max_len: int = 200):
        self.model_path = model_path
        self.max_len = max_len
        self._processor = None 

    @property
    def processor(self) -> sentencepiece.SentencePieceProcessor:
        if self._processor is None:
            self._processor = sentencepiece.SentencePieceProcessor(model_file=str(self.model_path))
        return self._processor

    def tokenize(self, task: str, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        task text + normalized state (23,) -> prompt_ids (max_len,) int64, prompt_mask (max_len,) bool.
        """
        text = task.strip().replace("_", " ").replace("\n", " ")
        edges = np.linspace(-1, 1, STATE_BINS + 1)[:-1]
        bins = np.digitize(state, bins=edges) - 1
        state_text = " ".join(str(b) for b in bins)
        prompt = f"Task: {text}, State: {state_text};\nAction: "

        tokens = self.processor.encode(prompt, add_bos=True)
        if len(tokens) > self.max_len:
            logger.warning("Prompt has %d tokens, truncating to %d", len(tokens), self.max_len)
            tokens = tokens[: self.max_len]

        prompt_ids = np.zeros(self.max_len, dtype=np.int64)  # 0 = padding token
        prompt_ids[: len(tokens)] = tokens
        prompt_mask = np.zeros(self.max_len, dtype=bool)
        prompt_mask[: len(tokens)] = True
        return prompt_ids, prompt_mask


def resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """
    Square uint8 image (H, W, 3) -> (size, size, 3), Pillow bilinear (as openpi).

    Used offline by prepare_dataset and live by the policy server, so training and
    inference images are resized identically.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 (H, W, 3), got {image.dtype} {image.shape}")

    height, width = image.shape[:2]
    if height != width:
        raise ValueError(f"expected a square image, got {height}x{width}")
    if height == size:
        return image
    return np.asarray(Image.fromarray(image).resize((size, size), Image.BILINEAR))


def prepare_image(image: torch.Tensor, train: bool, crop_and_rotate: bool) -> torch.Tensor:
    """
    uint8 (3, H, W) -> float32 (3, H, W) in [-1, 1]. Augments only when train is True.

    crop_and_rotate: True for the head camera, False for the wrist cameras (as openpi).
    """
    if image.ndim != 3 or image.shape[0] != 3 or image.dtype != torch.uint8:
        raise ValueError(f"expected uint8 (3, H, W), got {image.dtype} {tuple(image.shape)}")

    image = image.float() / 255.0  # [0, 1]
    if train:
        image = augment_image(image, crop_and_rotate)
    return image * 2.0 - 1.0


def uniform(low: float, high: float) -> float:
    return low + (high - low) * torch.rand(1).item()


def augment_image(image: torch.Tensor, crop_and_rotate: bool) -> torch.Tensor:
    """
    openpi's training augmentation on one float (3, H, W) image in [0, 1].

    Head camera: random 95% crop resized back, then rotation within +-5 degrees.
    All cameras: brightness, contrast, saturation. Clamped to [0, 1] at the end only.
    """
    _, height, width = image.shape

    if crop_and_rotate:
        crop_height, crop_width = int(height * 0.95), int(width * 0.95)
        top = torch.randint(0, height - crop_height + 1, (1,)).item()
        left = torch.randint(0, width - crop_width + 1, (1,)).item()
        image = image[:, top : top + crop_height, left : left + crop_width]
        image = F.interpolate(
            image[None], size=(height, width), mode="bilinear", align_corners=False
        )[0]

        angle = uniform(-5.0, 5.0)  # degrees
        if abs(angle) > 0.1:
            radians = math.radians(angle)
            cos, sin = math.cos(radians), math.sin(radians)
            rotation = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]])
            grid = F.affine_grid(rotation[None], [1, 3, height, width], align_corners=False)
            image = F.grid_sample(
                image[None], grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0]

    image = image * uniform(0.7, 1.3)  # brightness

    mean = image.mean()
    image = (image - mean) * uniform(0.6, 1.4) + mean  #  contrast

    gray = image.mean(dim=0, keepdim=True)  # plain channel average, as openpi
    image = gray + (image - gray) * uniform(0.5, 1.5)  # saturation

    return image.clamp(0.0, 1.0)