import torch

from helpers import TINY
from pi05.checkpoint import save_checkpoint
from pi05.models.pi05 import Pi05
from pi05.train import make_optimizer
from pi05.train_config import TrainConfig


def test_save_checkpoint_copies_extra_files(tmp_path):
    stats = tmp_path / "norm_stats.json"
    stats.write_text('{"state": {}}')
    config = TrainConfig(exp_name="test", model_config=TINY, checkpoint_base_dir=tmp_path)
    model = Pi05(TINY)

    step_dir = save_checkpoint(
        config.checkpoint_dir, 1, model, make_optimizer(model, config), config, (stats,)
    )
    assert (step_dir / "norm_stats.json").read_text() == '{"state": {}}'