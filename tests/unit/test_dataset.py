import numpy as np
from pathlib import Path

import pytest
import torch

from pi05.data_pipeline.dataset import B1kDataset, find_episode_and_frame, make_data_loader
from pi05.distributed import DistInfo

DEV_DATA = Path("prepared_data/dev")
TOKENIZER = Path("pretrained/paligemma_tokenizer.model")
needs_dev_data = pytest.mark.skipif(
    not (DEV_DATA.exists() and TOKENIZER.exists()), reason="prepared_data/dev or tokenizer missing"
)


def test_find_episode_at_boundaries():
    starts = np.array([0, 1956, 4166])  # episode lengths 1956, 2210, ...
    assert find_episode_and_frame(starts, 0) == (0, 0)  # first frame of the first episode
    assert find_episode_and_frame(starts, 1955) == (0, 1955)  # last frame of the first episode
    assert find_episode_and_frame(starts, 1956) == (1, 0)  # first frame of the second
    assert find_episode_and_frame(starts, 4200) == (2, 34)  # inside the third


@needs_dev_data
def test_sample_shapes_and_episode_end():
    dataset = B1kDataset(DEV_DATA, "train", TOKENIZER, augment=False)

    sample = dataset[0]
    assert sample["images"].shape == (3, 3, 224, 224)
    assert sample["images"].min() >= -1.0 and sample["images"].max() <= 1.0
    assert sample["prompt_ids"].shape == (200,)
    assert sample["actions"].shape == (50, 32)
    assert sample["action_mask"].all()  # frame 0: all 50 steps are real

    last_frame = int(dataset.starts[0] + dataset.lengths[0] - 1)  # first episode's last frame
    end = dataset[last_frame]
    assert end["action_mask"][0] and not end["action_mask"][1:].any()  # only step 0 is real
    assert torch.equal(end["actions"][1], end["actions"][0])  # the last action is repeated


@needs_dev_data
def test_data_loader_batches():
    dataset = B1kDataset(DEV_DATA, "train", TOKENIZER, augment=True)
    info = DistInfo(rank=0, local_rank=0, world_size=1, device=torch.device("cpu"))
    data = make_data_loader(dataset, batch_size=4, info=info, num_workers=2, seed=0)

    observation, actions, action_mask = next(data)
    assert observation.images.shape == (4, 3, 3, 224, 224)
    assert observation.prompt_ids.shape == (4, 200)
    assert actions.shape == (4, 50, 32)
    assert action_mask.shape == (4, 50)
    data.close()  # stop the loader: its worker processes shut down cleanly