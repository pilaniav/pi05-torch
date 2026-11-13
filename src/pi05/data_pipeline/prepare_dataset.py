"""
Prepare a training dataset from the raw download.

For every downloaded episode: copy its parquet unchanged, and re-encode its RGB videos
at 224x224 (H.264, keyframe every 10 frames, no B-frames) for fast frame-exact random
access. Frame counts are checked against the parquet and the episode length.

Output (out_dir) has the same layout as raw_dir, plus:
    meta/episodes.jsonl           prepared episodes only, each with "split" (train/val):
                                  per task, the last val_episodes_per_task episodes are val (as b1k)
    meta/info.json, tasks.jsonl   updated to describe the prepared files
    prepared_data_info.json       the settings used, and when
    norm_stats.json               exact stats of the 23-dim state and actions over train episodes

Interrupted or failed? Run the same command again: files already finished (and checked)
are skipped, and only the missing ones are made.

Changing image_size, keyframe_interval, crf or cameras? Use a new out_dir. Finished
files are never re-encoded, so reusing a folder would mix old and new settings.

Usage (from the repo root):
    # Everything downloaded, default settings
    uv run python -m pi05.data_pipeline.prepare_dataset

    # Small local test: 2 validation episodes per task, separate folder
    uv run python -m pi05.data_pipeline.prepare_dataset --out-dir prepared_data/dev --val-episodes-per-task 2

    # More parallel workers, delete the raw data afterwards
    uv run python -m pi05.data_pipeline.prepare_dataset --num-workers 8 --delete-raw
"""

import json
import logging
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from functools import partial
from pathlib import Path

import av
import pyarrow.parquet as pq
import torch
import tyro
from torchcodec.decoders import VideoDecoder

from pi05.data_pipeline.download import MANIFEST_FILE, read_jsonl
from pi05.data_pipeline.norm_stats import write_norm_stats
from pi05.data_pipeline.transforms import resize_image

logger = logging.getLogger(__name__)


FPS = 30


@dataclass(frozen=True)
class PrepareConfig:
    raw_dir: Path = Path("raw_data/behavior")  # relative to launch dir (repo root)
    out_dir: Path = Path("prepared_data/behavior")
    cameras: tuple[str, ...] = ("head", "left_wrist", "right_wrist")
    image_size: int = 224
    keyframe_interval: int = 10  # measured: ~1 ms random reads, ~3x smaller than every frame
    crf: int = 18   # H.264 quality: lower = better and bigger
    val_episodes_per_task: int = 10  # b1k: last 10 of 200 per task
    num_workers: int = 4  # episodes prepared in parallel
    delete_raw: bool = False  # delete raw_dir after success


def reencode_video(
    src: Path, dst: Path, size: int, keyframe_interval: int, crf: int = 18
) -> int:
    """
    Decode src frame by frame, resize to size x size, write H.264 to dst.

    Encoded for frame-exact random access: a keyframe every keyframe_interval frames,
    no B-frames (packets stay in display order), constant FPS. Returns the frame count.
    """
    with av.open(str(src)) as source, av.open(str(dst), mode="w") as target:
        source_stream = source.streams.video[0]
        # TODO: limit threads per worker (source_stream.thread_count = 2 and threads=2 in
        # x264-params). FFmpeg and x264 size their thread pools from the detected CPU count,
        # which in a container can be the whole host (e.g. 256 instead of 32 rented): with
        # many workers, thousands of threads compete for the cores. Measure prepare time first.
        source_stream.thread_type = "AUTO"  # decode with several threads

        target_stream = target.add_stream("libx264", rate=FPS)
        target_stream.width = size
        target_stream.height = size
        target_stream.pix_fmt = "yuv420p"
        target_stream.options = {
            "crf": str(crf),  # quality: lower = better and bigger; 18 ~ visually lossless
            # keyframe exactly every N frames, no extra ones at scene cuts, no B-frames
            "x264-params": (
                f"keyint={keyframe_interval}:min-keyint={keyframe_interval}"
                ":scenecut=0:bframes=0"
            ),
        }

        num_frames = 0
        for frame in source.decode(source_stream):
            if frame.width != frame.height:
                raise ValueError(f"{src}: expected square frames, got {frame.width}x{frame.height}")
            pixels = resize_image(frame.to_ndarray(format="rgb24"), size)

            new_frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")

            new_frame.pts = num_frames  # frame number = timestamp in units of 1/FPS s
            new_frame.time_base = Fraction(1, FPS)
            for packet in target_stream.encode(new_frame):
                target.mux(packet)
            num_frames += 1

        for packet in target_stream.encode():  # flush frames still inside the encoder
            target.mux(packet)

    return num_frames


def check_frame_access(path: Path, num_frames: int, num_checks: int = 5, seed: int = 0) -> None:
    """
    Check the video has num_frames frames, and that random access by index returns
    exactly the frames that reading in order does (as training will read them).
    """
    decoder = VideoDecoder(str(path))
    if len(decoder) != num_frames:
        raise ValueError(f"{path}: {len(decoder)} frames, expected {num_frames}")

    rng = random.Random(seed)
    indices = rng.sample(range(num_frames), min(num_checks, num_frames))

    random_frames = {}
    for index in indices:  # random order: forces real seeks
        random_frames[index] = decoder[index]

    in_order = VideoDecoder(str(path))
    for index in range(max(indices) + 1):  # read in order up to the last checked frame
        frame = in_order[index]
        if index in random_frames and not torch.equal(frame, random_frames[index]):
            raise ValueError(f"{path}: frame {index} differs between random and in-order reads")


def episode_paths(
    root: Path, info: dict, episode_index: int, cameras: tuple[str, ...]
) -> tuple[Path, dict[str, Path]]:
    """
    Parquet path and per-camera video paths of one episode under root.

    Same templates (from info.json) for raw_data and prepared_data: the layout is identical.
    """
    chunk = episode_index // info["chunks_size"]  # chunk is basically task_index
    parquet = root / info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)
    videos = {}
    for camera in cameras:
        videos[camera] = root / info["video_path"].format(
            episode_chunk=chunk,
            video_key=f"observation.images.rgb.{camera}",
            episode_index=episode_index,
        )
    return parquet, videos


def prepare_episode(
    episode_index: int,
    length: int,
    raw_dir: Path,
    out_dir: Path,
    info: dict,
    cameras: tuple[str, ...],
    size: int,
    keyframe_interval: int,
    crf: int
) -> None:
    """
    Copy one episode's parquet and re-encode its videos at size x size, with checks.

    Every file is written under a temporary name and renamed only once complete and
    checked, so a file at its final path is always good: re-running skips it.
    """
    raw_parquet, raw_videos = episode_paths(raw_dir, info, episode_index, cameras)
    out_parquet, out_videos = episode_paths(out_dir, info, episode_index, cameras)

    if not out_parquet.exists():
        num_rows = pq.ParquetFile(raw_parquet).metadata.num_rows
        if num_rows != length:
            raise ValueError(f"episode {episode_index}: {num_rows} parquet rows, expected {length}")
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = out_parquet.with_name(out_parquet.name + ".partial")
        shutil.copyfile(raw_parquet, tmp_file)
        tmp_file.replace(out_parquet)

    for camera in cameras:
        out_video = out_videos[camera]
        if out_video.exists():
            continue
        out_video.parent.mkdir(parents=True, exist_ok=True)
        tmp_file = out_video.with_name(out_video.stem + ".partial.mp4")  # .mp4 so PyAV knows the format

        num_frames = reencode_video(raw_videos[camera], tmp_file, size, keyframe_interval, crf)
        if num_frames != length:
            raise ValueError(
                f"episode {episode_index}, {camera}: {num_frames} frames, expected {length}"
            )
        check_frame_access(tmp_file, num_frames)
        tmp_file.replace(out_video)


def write_text_atomic(path: Path, text: str) -> None:
    """
    Write via a temporary file + rename, so a crash never leaves a half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = path.with_name(path.name + ".partial")
    tmp_file.write_text(text)
    tmp_file.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    lines = []
    for row in rows:
        lines.append(json.dumps(row))
    write_text_atomic(path, "\n".join(lines) + "\n")


def assign_splits(episodes: list[dict], chunks_size: int, val_per_task: int) -> list[dict]:
    """
    Copies of episodes with "split" added: per task, the last val_per_task (by id) are "val".
    """
    by_task = {}
    for episode in sorted(episodes, key=lambda e: e["episode_index"]):
        chunk = episode["episode_index"] // chunks_size  # chunk is basically task_index
        by_task.setdefault(chunk, []).append(episode)

    result = []
    for chunk, task_episodes in by_task.items():
        num_train = len(task_episodes) - val_per_task
        if num_train < 1:
            raise ValueError(
                f"task {chunk} has {len(task_episodes)} episodes: too few for {val_per_task} val"
            )
        for position, episode in enumerate(task_episodes):
            split = "train" if position < num_train else "val"
            result.append({**episode, "split": split})
    return result


def write_metadata(
    raw_dir: Path,
    out_dir: Path,
    info: dict,
    episodes: list[dict],
    cameras: tuple[str, ...],
    size: int,
    prepare_info: dict,
) -> None:
    """
    Write meta/ (info, tasks, episodes with split) and prepared_data_info.json.
    """
    # info.json: keep only the RGB cameras we prepared, at their new size and codec.
    new_info = json.loads(json.dumps(info))  # deep copy
    for key in list(new_info["features"]):
        if key.startswith("observation.images."):
            camera = key.removeprefix("observation.images.rgb.")
            if not key.startswith("observation.images.rgb.") or camera not in cameras:
                del new_info["features"][key]
                continue
            feature = new_info["features"][key]
            feature["shape"] = [size, size, 3]
            if "info" in feature:
                feature["info"]["video.height"] = size
                feature["info"]["video.width"] = size
                feature["info"]["video.codec"] = "h264"
    new_info["total_episodes"] = len(episodes)
    new_info["total_frames"] = sum(episode["length"] for episode in episodes)
    write_text_atomic(out_dir / "meta/info.json", json.dumps(new_info, indent=2))

    # tasks.jsonl: only the tasks that have prepared episodes.
    chunks = set()
    for episode in episodes:
        chunks.add(episode["episode_index"] // info["chunks_size"])
    tasks = []
    for line in (raw_dir / "meta/tasks.jsonl").read_text().splitlines():
        if line.strip():
            task = json.loads(line)
            if task["task_index"] in chunks:
                tasks.append(task)
    write_jsonl(out_dir / "meta/tasks.jsonl", tasks)

    write_jsonl(out_dir / "meta/episodes.jsonl", episodes)

    prepare_info["created"] = datetime.now(UTC).isoformat()
    write_text_atomic(out_dir / "prepared_data_info.json", json.dumps(prepare_info, indent=2))


def prepare_one(record: dict, config: PrepareConfig, info: dict) -> None:
    prepare_episode(
        record["episode_index"], record["length"], config.raw_dir, config.out_dir, info,
        config.cameras, config.image_size, config.keyframe_interval, config.crf,
    )


def main() -> None:
    config = tyro.cli(PrepareConfig)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    info = json.loads((config.raw_dir / "meta/info.json").read_text())

    # Episodes on disk, from the download manifest.
    manifest = json.loads((config.raw_dir / MANIFEST_FILE).read_text())
    downloaded_ids = set()
    for task_episodes in manifest["episodes"].values():
        for episode_key in task_episodes:
            downloaded_ids.add(int(episode_key))
    downloaded_records = []
    for record in read_jsonl(config.raw_dir / "meta/episodes.jsonl"):
        if record["episode_index"] in downloaded_ids:
            downloaded_records.append(record)
    split_records = assign_splits(
        downloaded_records, info["chunks_size"], config.val_episodes_per_task
    )

    logger.info("Preparing %d episodes with %d workers", len(split_records), config.num_workers)
    with ProcessPoolExecutor(max_workers=config.num_workers) as pool:
        list(pool.map(partial(prepare_one, config=config, info=info), split_records))

    prepare_info = json.loads(json.dumps(asdict(config), default=str))
    write_metadata(
        config.raw_dir, config.out_dir, info, split_records, config.cameras, config.image_size,
        prepare_info
    )

    write_norm_stats(config.out_dir)

    if config.delete_raw:
        shutil.rmtree(config.raw_dir)
    logger.info("Done: %s", config.out_dir.resolve())


if __name__ == "__main__":
    main()