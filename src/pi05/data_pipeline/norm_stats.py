"""
Normalization statistics of a prepared dataset: exact, over all train frames.

Called by prepare_dataset at the end; can also run alone (e.g. after --delete-raw):
    uv run python -m pi05.data_pipeline.norm_stats --dataset-dir prepared_data/behavior
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from pi05.data_pipeline.b1k_state_action import extract_state, read_state_actions
from pi05.data_pipeline.download import read_jsonl


logger = logging.getLogger(__name__)


NORM_STATS_FILE = "norm_stats.json"


def describe(values: np.ndarray) -> dict[str, list[float]]:
    """
    Per-dimension statistics of (N, D) values.
    """
    values = values.astype(np.float64)  # accurate sums over ~1M frames
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def write_norm_stats(dataset_dir: Path) -> None:
    """
    Compute stats of the 23-dim state and actions over the train split; save them.
    """
    info = json.loads((dataset_dir / "meta/info.json").read_text())
    states = []
    actions = []
    for record in read_jsonl(dataset_dir / "meta/episodes.jsonl"):
        if record["split"] != "train":
            continue
        episode_index = record["episode_index"]
        parquet = dataset_dir / info["data_path"].format(
            episode_chunk=episode_index // info["chunks_size"],  # chunk is basically task_index
            episode_index=episode_index,
        )
        raw_state, episode_actions = read_state_actions(parquet)
        states.append(extract_state(raw_state))
        actions.append(episode_actions)

    all_states = np.concatenate(states)
    all_actions = np.concatenate(actions)
    stats = {
        "num_frames": len(all_states),
        "state": describe(all_states),
        "actions": describe(all_actions),
    }
    (dataset_dir / NORM_STATS_FILE).write_text(json.dumps(stats, indent=2))
    logger.info("Norm stats over %d train frames: %s", len(all_states), dataset_dir / NORM_STATS_FILE)


@dataclass(frozen=True)
class NormStatsConfig:
    dataset_dir: Path = Path("prepared_data/behavior")


def main() -> None:
    config = tyro.cli(NormStatsConfig)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    write_norm_stats(config.dataset_dir)


if __name__ == "__main__":
    main()