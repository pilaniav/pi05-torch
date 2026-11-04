import pytest
from fractions import Fraction

import av
import numpy as np
from torchcodec.decoders import VideoDecoder

from pi05.data_pipeline.prepare_dataset import FPS, check_frame_access, reencode_video
from pi05.data_pipeline.prepare_dataset import assign_splits


def make_video(path, num_frames, size):
    """Video whose frame i is uniformly gray with value 5 * i (frame index readable from pixels)."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width = size
        stream.height = size
        stream.pix_fmt = "yuv420p"
        stream.options = {"x264-params": "keyint=250"}  # long keyframe interval, like the raw data
        for i in range(num_frames):
            pixels = np.full((size, size, 3), 5 * i, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_reencoded_frames_are_exact_by_index(tmp_path):
    src, dst = tmp_path / "src.mp4", tmp_path / "dst.mp4"
    make_video(src, num_frames=40, size=64)

    num_frames = reencode_video(src, dst, size=32, keyframe_interval=5)
    assert num_frames == 40
    check_frame_access(dst, num_frames, num_checks=10)

    decoder = VideoDecoder(str(dst))
    for i in [37, 3, 20, 0, 39]:  # random order
        brightness = decoder[i].float().mean().item()
        assert abs(brightness - 5 * i) < 3, f"frame {i} has brightness {brightness}"


def test_assign_splits_last_episodes_of_each_task_are_val():
    episodes = []
    for episode_index in [30, 10, 20, 10020, 10010, 10030]:  # 2 tasks, unsorted on purpose
        episodes.append({"episode_index": episode_index, "length": 100})

    result = assign_splits(episodes, chunks_size=10_000, val_per_task=1)

    splits = {}
    for episode in result:
        splits[episode["episode_index"]] = episode["split"]
    assert splits == {
        10: "train", 20: "train", 30: "val",
        10010: "train", 10020: "train", 10030: "val",
    }
    with pytest.raises(ValueError):
        assign_splits(episodes, chunks_size=10_000, val_per_task=3)  # no train episodes left