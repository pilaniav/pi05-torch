import numpy as np

from pi05.data_pipeline.b1k_state_action import STATE_DIM, extract_state


def test_extract_state_indices_and_order():
    raw = np.arange(256, dtype=np.float32)  # value == index
    expected = (
        [253, 254, 255]                          # base_qvel
        + [236, 237, 238, 239]                   # trunk
        + list(range(158, 165))                  # left arm
        + list(range(197, 204))                  # right arm
        + [193 + 194, 232 + 233]                 # gripper widths (finger sums)
    )
    state = extract_state(raw)
    assert state.shape == (STATE_DIM,)
    assert state.tolist() == expected
    assert extract_state(np.stack([raw, raw])).shape == (2, STATE_DIM)  # batched