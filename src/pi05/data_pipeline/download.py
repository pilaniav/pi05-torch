"""
Download BEHAVIOR-1K demos: per episode, the parquet and the RGB videos.

Re-running the same command resumes: complete files are skipped, failed ones retried.
After each run, download_manifest.json in raw_dir lists everything on disk.

Rate limits: Hugging Face answers HTTP 429 when there are too many requests. Those
are retried with growing waits (up to ~5 min per file). Files that still fail are
listed at the end; re-run the same command to fetch them. Logging in
(`huggingface-cli login`) gives higher limits; fewer --num-workers also helps.

Usage (from the repo root):
    # Default tasks (turning_on_radio, picking_up_trash), all episodes
    uv run python -m pi05.data_pipeline.download

    # Default tasks, 2 episodes each (quick local development)
    uv run python -m pi05.data_pipeline.download --episodes-per-task 2

    # One task, a few episodes
    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio --episodes-per-task 5

    # Several chosen tasks, all episodes
    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio picking_up_trash

    # Every task, 1 episode each (survey the whole dataset cheaply)
    uv run python -m pi05.data_pipeline.download --all-tasks --episodes-per-task 1

    # Every task, every episode
    uv run python -m pi05.data_pipeline.download --all-tasks

    # Only some cameras
    uv run python -m pi05.data_pipeline.download --cameras head --episodes-per-task 2

    # Different location (e.g. a mounted disk on vast.ai), more parallel downloads
    uv run python -m pi05.data_pipeline.download --raw-dir /workspace/raw_data/behavior --num-workers 16

    # Delete everything already downloaded and start over
    uv run python -m pi05.data_pipeline.download --fresh
"""

import json
import logging
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import tyro
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

logger = logging.getLogger(__name__)


META_FILES = ["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"]
MANIFEST_FILE = "download_manifest.json"
PERMANENT_ERRORS = (EntryNotFoundError, GatedRepoError, RepositoryNotFoundError)


@dataclass(frozen=True)
class DownloadConfig:
    raw_dir: Path = Path("raw_data/behavior")  # relative to launch dir (repo root)
    repo_id: str = "behavior-1k/2025-challenge-demos"  # Hugging Face dataset
    tasks: tuple[str, ...] = ("turning_on_radio", "picking_up_trash")
    all_tasks: bool = False  # ignore `tasks` and download every task
    episodes_per_task: int | None = None  # None = all; a few for development
    cameras: tuple[str, ...] = ("head", "left_wrist", "right_wrist")  # RGB cameras
    fresh: bool = False  # delete raw_dir and start over
    num_workers: int = 8  # parallel downloads
    max_retries: int = 5


def download_file(filename: str, config: DownloadConfig) -> None:
    """
    Download one repo file into raw_dir; retry transient errors with growing waits.
    """
    for attempt in range(1, config.max_retries + 1):
        try:
            hf_hub_download(
                config.repo_id, filename, repo_type="dataset", local_dir=config.raw_dir
            )
            return
        except PERMANENT_ERRORS:
            raise
        except Exception as error:  # network errors, rate limits
            if attempt == config.max_retries:
                raise
            wait_seconds = 2**attempt
            wait_seconds = min(5 * 2**attempt, 300)  # long enough to outlast HF rate-limit windows
            logger.warning("%s failed (%s); retry %d in %ds", filename, error, attempt, wait_seconds)
            time.sleep(wait_seconds)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def rgb_cameras(info: dict) -> list[str]:
    """
    Names of all RGB cameras in the dataset, from info.json's features.
    """
    cameras = []
    for key in info["features"]:
        if key.startswith("observation.images.rgb."):
            cameras.append(key.removeprefix("observation.images.rgb."))
    return cameras


def check_cameras(cameras: tuple[str, ...], info: dict) -> None:
    """
    Fail early if a camera has no RGB video in this dataset.
    """
    available = rgb_cameras(info)
    for camera in cameras:
        if camera not in available:
            raise ValueError(f"Unknown camera {camera!r}. Available: {available}")


def select_episodes(config: DownloadConfig, chunks_size: int) -> dict[str, list[int]]:
    """
    Task name -> sorted episode indices (optionally only the first N per task).
    """
    task_index_by_name = {}
    for task in read_jsonl(config.raw_dir / "meta/tasks.jsonl"):
        task_index_by_name[task["task_name"]] = task["task_index"]

    names = list(task_index_by_name) if config.all_tasks else list(config.tasks)
    unknown = []
    for name in names:
        if name not in task_index_by_name:
            unknown.append(name)
    if unknown:
        raise ValueError(f"Unknown tasks {unknown}. Available: {sorted(task_index_by_name)}")

    # b1k stores one task per chunk: episode_index // chunks_size = task index.
    episodes_by_task_index = {}
    for episode in read_jsonl(config.raw_dir / "meta/episodes.jsonl"):
        task_index = episode["episode_index"] // chunks_size
        episodes_by_task_index.setdefault(task_index, []).append(episode["episode_index"])

    selected = {}
    for name in names:
        episodes = sorted(episodes_by_task_index.get(task_index_by_name[name], []))
        if config.episodes_per_task is not None:
            episodes = episodes[: config.episodes_per_task]
        if not episodes:
            raise ValueError(f"No episodes found for task {name!r}")
        logger.info("%s: %d episodes", name, len(episodes))
        selected[name] = episodes

    num_episodes = sum(len(episodes) for episodes in selected.values())
    logger.info("%d tasks, %d episodes", len(selected), num_episodes)
    return selected


def episode_files(episode_index: int, info: dict, cameras: tuple[str, ...]) -> list[str]:
    """
    Repo paths of one episode's parquet and RGB videos, from info.json's templates.
    """
    chunk = episode_index // info["chunks_size"]
    files = [info["data_path"].format(episode_chunk=chunk, episode_index=episode_index)]
    for camera in cameras:
        video_key = f"observation.images.rgb.{camera}"
        files.append(
            info["video_path"].format(
                episode_chunk=chunk, video_key=video_key, episode_index=episode_index
            )
        )
    return files


def download_all(files: list[str], config: DownloadConfig) -> list[str]:
    """
    Download files in parallel. Returns the files that still failed after retries.
    """
    failed = []
    with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
        future_to_file = {}
        for filename in files:
            future_to_file[pool.submit(download_file, filename, config)] = filename

        for num_done, future in enumerate(as_completed(future_to_file), start=1):
            filename = future_to_file[future]
            try:
                future.result()  # re-raises the download's error, if any
            except Exception as error:  # noqa: BLE001 - any failure: record it, keep going
                logger.error("Giving up on %s: %s", filename, error)
                failed.append(filename)
            if num_done % 100 == 0 or num_done == len(files):
                logger.info("%d / %d files", num_done, len(files))
    return failed


def build_manifest(config: DownloadConfig, info: dict) -> dict:
    """Describe everything on disk: task -> episode -> cameras with a video."""
    task_name_by_index = {}
    for task in read_jsonl(config.raw_dir / "meta/tasks.jsonl"):
        task_name_by_index[task["task_index"]] = task["task_name"]

    all_cameras = rgb_cameras(info)  # computed once, used for every episode below

    episodes = {}
    for parquet in sorted(config.raw_dir.glob("data/task-*/episode_*.parquet")):
        episode_index = int(parquet.stem.removeprefix("episode_"))
        chunk = episode_index // info["chunks_size"]  # chunk is basically task_index

        cameras = []
        for camera in all_cameras:
            video = config.raw_dir / info["video_path"].format(
                episode_chunk=chunk,
                video_key=f"observation.images.rgb.{camera}",
                episode_index=episode_index,
            )
            if video.exists():
                cameras.append(camera)

        task_episodes = episodes.setdefault(task_name_by_index[chunk], {})
        task_episodes[str(episode_index)] = cameras

    return {"repo_id": config.repo_id, "episodes": episodes}


def check_same_repo(config: DownloadConfig) -> None:
    """
    Refuse to mix two datasets in one raw_dir.
    """
    manifest_path = config.raw_dir / MANIFEST_FILE
    if manifest_path.exists():
        existing_repo = json.loads(manifest_path.read_text())["repo_id"]
        if existing_repo != config.repo_id:
            raise ValueError(
                f"{config.raw_dir} holds {existing_repo}, not {config.repo_id}: use another raw_dir"
            )


def write_manifest(config: DownloadConfig, info: dict) -> None:
    """
    Write the manifest atomically: a crash never leaves a half-written file.
    """
    manifest_path = config.raw_dir / MANIFEST_FILE
    tmp_path = manifest_path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(build_manifest(config, info), indent=2))
    tmp_path.replace(manifest_path)
    logger.info("Manifest (everything on disk): %s", manifest_path)


def main() -> None:
    config = tyro.cli(DownloadConfig)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if config.fresh and config.raw_dir.exists():
        logger.info("Deleting %s", config.raw_dir)
        shutil.rmtree(config.raw_dir)
    check_same_repo(config)

    # Metadata first and one at a time: a gated or missing repo fails here with one clear error.
    for filename in META_FILES:
        download_file(filename, config)
    info = json.loads((config.raw_dir / "meta/info.json").read_text())
    check_cameras(config.cameras, info)

    selected = select_episodes(config, info["chunks_size"])
    files = []
    for episodes in selected.values():
        for episode_index in episodes:
            files.extend(episode_files(episode_index, info, config.cameras))

    logger.info("Downloading %d files to %s", len(files), config.raw_dir.resolve())
    failed = download_all(files, config)
    write_manifest(config, info)          # always: reflects what's on disk now

    if failed:
        logger.error("%d files failed. Re-run the same command to retry them.", len(failed))
        sys.exit(1)
    logger.info("Done")


if __name__ == "__main__":
    main()