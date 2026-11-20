"""
Inference policy: one observation in, one action chunk out.

Policy holds the trained model and applies the same input processing as training
(no augmentation) and the inverse normalization on the output. It knows nothing
about the network or the BEHAVIOR eval protocol; the wrapper and the server do.
"""


from collections import deque
from pathlib import Path

import numpy as np
import torch

from pi05.checkpoint import load_pretrained
from pi05.data_pipeline.b1k_state_action import extract_state
from pi05.data_pipeline.norm_stats import NORM_STATS_FILE
from pi05.data_pipeline.transforms import (
    PromptTokenizer,
    load_norm_stats,
    normalize,
    prepare_image,
    resize_image,
    unnormalize,
)
from pi05.models.observation import Observation
from pi05.models.pi05 import Pi05
from pi05.models.pi05_config import Pi05Config

IMAGE_SIZE = 224  # the size the model was trained on
NUM_CAMERAS = 3  # head, left wrist, right wrist (same order as training)

# BEHAVIOR eval observation keys (flattened as parent::child by their eval).
PROPRIO_KEY = "robot_r1::proprio"
CAMERA_KEYS = (  # same order as training: head, left wrist, right wrist
    "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
)
# Action dims of the two grippers (left, right) in R1Pro's 23-dim action.
GRIPPER_DIMS = (14, 22)
ENSEMBLE_WEIGHT_RATE = 0.005  # BEHAVIOR's baseline: weights exp(rate * i), newer slightly heavier


class Policy:
    """
    Trained pi0.5 for inference: (images, proprio, prompt) -> action chunk.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        model_config: Pi05Config,
        tokenizer_path: Path,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        checkpoint_dir holds model.safetensors and the training run's norm_stats.json.
        dtype: bfloat16 on GPU (~6.7 GB of weights); float32 for CPU tests.
        """
        self.device = device
        self.dtype = dtype

        model = Pi05(model_config)
        load_pretrained(model, checkpoint_dir / "model.safetensors")
        model.to(device=device, dtype=dtype)
        model.eval()
        self.model = model

        # The training run's stats: the model learned on data normalized with exactly these.
        self.norm_stats = load_norm_stats(checkpoint_dir / NORM_STATS_FILE)
        self.tokenizer = PromptTokenizer(tokenizer_path, max_len=model_config.max_token_len)

    def infer(self, images: list[np.ndarray], proprio: np.ndarray, prompt: str) -> np.ndarray:
        """
        images:  NUM_CAMERAS RGB images, uint8 (H, W, 3), in training camera order.
        proprio: BEHAVIOR's raw robot state, (256,).
        prompt:  task text.
        Returns the action chunk, unnormalized: (action_horizon, robot action dims), float32.
        """
        if len(images) != NUM_CAMERAS:
            raise ValueError(f"expected {NUM_CAMERAS} images, got {len(images)}")

        # State: same extraction and normalization as training; it goes into the prompt.
        state = normalize(extract_state(proprio), self.norm_stats["state"])
        prompt_ids, prompt_mask = self.tokenizer.tokenize(prompt, state)

        # Images: resize (a no-op at 224), HWC -> CHW, uint8 -> [-1, 1], no augmentation.
        prepared = []
        for image in images:
            image = resize_image(image, IMAGE_SIZE)
            chw = torch.from_numpy(image).permute(2, 0, 1).contiguous()
            prepared.append(prepare_image(chw, train=False, crop_and_rotate=False))

        obs = Observation(
            images=torch.stack(prepared)[None],  # (1, cams, 3, H, W)
            image_masks=torch.ones(1, NUM_CAMERAS, dtype=torch.bool),
            prompt_ids=torch.from_numpy(prompt_ids)[None],  # (1, L)
            prompt_mask=torch.from_numpy(prompt_mask)[None],  # (1, L)
        ).to(self.device)

        # Same autocast as training; off for float32 (CPU tests).
        use_autocast = self.dtype == torch.bfloat16
        with torch.inference_mode(), torch.autocast(
            self.device.type, dtype=torch.bfloat16, enabled=use_autocast
        ):
            actions = self.model.sample_actions(obs)  # (1, horizon, 32), normalized, padded

        # Drop the padding to the robot's dims, then undo the normalization.
        robot_dims = self.norm_stats["actions"]["q01"].shape[-1]  # 23 for R1Pro
        chunk = actions[0, :, :robot_dims].float().cpu().numpy()
        return unnormalize(chunk, self.norm_stats["actions"])


class B1KPolicy:
    """
    Adapts Policy to the BEHAVIOR eval: one observation per sim step in, one action out.

    The eval sends BEHAVIOR's flattened observation dict every sim step and executes
    exactly one action per call, so chunk handling lives here (server side):

    temporal_ensemble: call the model every step and average the current step's
                       predictions from the last ensemble_size chunks (BEHAVIOR's 
                       baseline default).
    receding_horizon: call the model, execute replan_steps actions, then call again.
    """

    def __init__(
        self,
        policy: Policy,
        prompt: str,
        mode: str = "temporal_ensemble",
        ensemble_size: int = 10,
        replan_steps: int = 10,
    ):
        if mode not in ("temporal_ensemble", "receding_horizon"):
            raise ValueError(f"unknown mode: {mode}")
        self.policy = policy
        self.prompt = prompt  # fixed: the eval sends no prompt, so one server = one task
        self.mode = mode
        self.ensemble_size = ensemble_size
        self.replan_steps = replan_steps
        self.reset()

    def reset(self) -> None:
        """
        Forget all chunks: called at the start of every task instance.
        """
        self.chunks = deque(maxlen=self.ensemble_size)  # temporal_ensemble, newest last
        self.plan = None  # receding_horizon: current chunk
        self.plan_step = 0

    def act(self, obs: dict) -> np.ndarray:
        """
        obs: BEHAVIOR's observation dict for the current sim step.
        Returns one action, (1, robot action dims) float32.
        """
        if self.mode == "temporal_ensemble":
            self.chunks.append(self._infer(obs))
            action = self._ensemble_action()
        else:
            if self.plan is None or self.plan_step >= self.replan_steps:
                self.plan = self._infer(obs)
                self.plan_step = 0
            action = self.plan[self.plan_step]
            self.plan_step += 1
        return action[None].astype(np.float32)

    def _infer(self, obs: dict) -> np.ndarray:
        """
        Pick our inputs from BEHAVIOR's obs and get a chunk (horizon, action dims).
        """
        images = []
        for key in CAMERA_KEYS:
            image = obs[key]
            if image.dtype != np.uint8:
                raise ValueError(f"{key}: expected uint8 image, got {image.dtype}")
            # Drop alpha (RGBA -> RGB); np.array copies, since msgpack arrays are read-only.
            images.append(np.array(image[..., :3]))
        proprio = np.asarray(obs[PROPRIO_KEY], dtype=np.float32)
        return self.policy.infer(images, proprio, self.prompt)

    def _ensemble_action(self) -> np.ndarray:
        """
        Weighted average of every stored chunk's prediction for the current step.

        A chunk made `age` steps ago predicts the current step at index `age`.
        Grippers come from the newest chunk only: averaging open and close would give
        a half-open command.
        """
        num_chunks = len(self.chunks)
        predictions = []
        for i, chunk in enumerate(self.chunks):  # oldest first
            age = num_chunks - 1 - i
            predictions.append(chunk[age])
        predictions = np.stack(predictions)  # (num_chunks, action dims)

        weights = np.exp(ENSEMBLE_WEIGHT_RATE * np.arange(num_chunks))  # newer slightly heavier
        weights = weights / weights.sum()
        action = (weights[:, None] * predictions).sum(axis=0)

        newest = self.chunks[-1][0]
        for dim in GRIPPER_DIMS:
            action[dim] = newest[dim]
        return action