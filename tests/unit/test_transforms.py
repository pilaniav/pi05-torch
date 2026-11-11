import numpy as np
import pytest
from pathlib import Path

import torch

from pi05.data_pipeline.transforms import normalize, pad_last_dim, unnormalize
from pi05.data_pipeline.transforms import PromptTokenizer
from pi05.data_pipeline.transforms import augment_image, prepare_image, resize_image


STATS = {"q01": np.array([0.0, -2.0], np.float32), "q99": np.array([1.0, 2.0], np.float32)}


TOKENIZER = Path("pretrained/paligemma_tokenizer.model")
needs_tokenizer = pytest.mark.skipif(not TOKENIZER.exists(), reason="tokenizer file not downloaded")


def test_normalize_maps_quantiles_and_inverts():
    x = np.array([[0.0, -2.0], [1.0, 2.0], [0.25, 0.5]], np.float32)  # q01 row, q99 row, other
    y = normalize(x, STATS)
    assert y[0] == pytest.approx([-1.0, -1.0], abs=1e-5)  # q01 -> -1
    assert y[1] == pytest.approx([1.0, 1.0], abs=1e-5)  # q99 -> +1
    assert unnormalize(y, STATS) == pytest.approx(x, abs=1e-5)  # round trip


def test_pad_last_dim():
    x = np.ones((50, 23), np.float32)
    padded = pad_last_dim(x, 32)
    assert padded.shape == (50, 32)
    assert np.array_equal(padded[:, :23], x)
    assert np.all(padded[:, 23:] == 0)


@needs_tokenizer
def test_prompt_matches_openpi_format():
    tokenizer = PromptTokenizer(TOKENIZER, max_len=200)
    state = np.array([-1.0, 0.0, 0.999, -1.5, 2.0], np.float32)  # bins 0, 128, 255, -1, 255
    ids, mask = tokenizer.tokenize("turn_on_radio\n", state)

    expected = "Task: turn on radio, State: 0 128 255 -1 255;\nAction: "
    expected_ids = tokenizer.processor.encode(expected, add_bos=True)
    num_tokens = len(expected_ids)
    assert ids[:num_tokens].tolist() == expected_ids
    assert mask.sum() == num_tokens
    assert np.all(ids[num_tokens:] == 0)
    assert ids.dtype == np.int64


@needs_tokenizer
def test_prompt_truncates_to_max_len():
    tokenizer = PromptTokenizer(TOKENIZER, max_len=5)
    ids, mask = tokenizer.tokenize("turn on the radio", np.zeros(23, np.float32))
    assert ids.shape == (5,)
    assert mask.all()


def test_prepare_image_scales_to_minus_one_one():
    image = torch.zeros(3, 4, 4, dtype=torch.uint8)
    image[:, 0, 0] = 255
    out = prepare_image(image, train=False, crop_and_rotate=True)
    assert out.dtype == torch.float32
    assert out[0, 0, 0] == 1.0  # 255 -> 1
    assert out[0, 1, 1] == -1.0  # 0 -> -1


def test_augment_image_keeps_shape_and_range_and_changes_pixels():
    torch.manual_seed(0)
    image = torch.rand(3, 224, 224)
    for crop_and_rotate in [True, False]:
        out = augment_image(image, crop_and_rotate)
        assert out.shape == (3, 224, 224)
        assert out.min() >= 0.0 and out.max() <= 1.0
        assert not torch.equal(out, image)


def test_resize_image():
    image = np.zeros((720, 720, 3), np.uint8)
    assert resize_image(image, 224).shape == (224, 224, 3)
    with pytest.raises(ValueError):
        resize_image(np.zeros((720, 480, 3), np.uint8), 224)