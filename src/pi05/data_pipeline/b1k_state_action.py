"""
BEHAVIOR-1K (R1Pro robot) specifics: parquet columns and the 23-dim state the model sees.

Indices from BEHAVIOR-1K's PROPRIOCEPTION_INDICES["R1Pro"] (eval_utils.py) and openpi's
extract_state_from_proprio (b1k_policy.py). The order must match theirs exactly.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

STATE_COLUMN = "observation.state"  # raw proprioception, 256 values per frame
ACTION_COLUMN = "action"  # 23 values: base 3, torso 4, arm L 7, gripper L 1, arm R 7, gripper R 1


# Parts of the raw state, concatenated in this order (b1k's: both arms, then both grippers).
STATE_PARTS = [
    slice(253, 256),  # base_qvel (a velocity)
    slice(236, 240),  # trunk_qpos
    slice(158, 165),  # arm_left_qpos
    slice(197, 204),  # arm_right_qpos
]
GRIPPER_FINGERS = [
    slice(193, 195),  # gripper_left_qpos: 2 fingers, summed into one width
    slice(232, 234),  # gripper_right_qpos
]
STATE_DIM = 23



def extract_state(raw_state: np.ndarray) -> np.ndarray:
    """
    (..., 256) raw proprioception -> (..., 23) state, exactly as b1k.
    """
    parts = []
    for part in STATE_PARTS:
        parts.append(raw_state[..., part])
    for fingers in GRIPPER_FINGERS:
        parts.append(raw_state[..., fingers].sum(axis=-1, keepdims=True))
    return np.concatenate(parts, axis=-1).astype(np.float32)


def read_state_actions(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    One episode's parquet -> raw state (T, 256) and actions (T, 23), float32.
    """
    table = pq.read_table(parquet, columns=[STATE_COLUMN, ACTION_COLUMN])
    raw_state = np.stack(table[STATE_COLUMN].to_numpy())
    actions = np.stack(table[ACTION_COLUMN].to_numpy())
    return raw_state.astype(np.float32), actions.astype(np.float32)