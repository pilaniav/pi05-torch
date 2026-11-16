import torch

from helpers import TINY
from pi05.checkpoint import load_pretrained, save_checkpoint
from pi05.models.pi05 import Pi05
from pi05.train import make_optimizer
from pi05.train_config import get_train_config, TrainConfig


def test_save_checkpoint_copies_extra_files(tmp_path):
    stats = tmp_path / "norm_stats.json"
    stats.write_text('{"state": {}}')
    config = TrainConfig(exp_name="test", model_config=TINY, checkpoint_base_dir=tmp_path)
    model = Pi05(TINY)

    step_dir = save_checkpoint(
        config.checkpoint_dir, 1, model, make_optimizer(model, config), config, (stats,)
    )
    assert (step_dir / "norm_stats.json").read_text() == '{"state": {}}'


def test_load_pretrained_reads_our_own_checkpoint(tmp_path):
    """
    Weights saved by save_checkpoint load into a fresh model with our names as-is.
    """
    config = get_train_config("debug")

    torch.manual_seed(0)
    trained = Pi05(config.model_config)
    optimizer = make_optimizer(trained, config)
    step_dir = save_checkpoint(tmp_path, 1, trained, optimizer, config)

    torch.manual_seed(1)  # different init, so a load that did nothing would fail below
    fresh = Pi05(config.model_config)
    load_pretrained(fresh, step_dir / "model.safetensors")

    expected = trained.state_dict()
    for name, tensor in fresh.state_dict().items():
        torch.testing.assert_close(tensor, expected[name], msg=name)