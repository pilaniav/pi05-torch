import pytest
import tyro

from pi05.train_config import TRAIN_CONFIGS


@pytest.mark.parametrize("name", list(TRAIN_CONFIGS))
def test_cli_selects_config_and_overrides(name):
    choices = {}
    for config_name, config in TRAIN_CONFIGS.items():
        choices[config_name] = (config_name, config)

    config = tyro.extras.overridable_config_cli(
        choices, args=[name, "--num-train-steps", "7", "--model-config.action-horizon", "3"]
    )

    assert config.exp_name == name
    assert config.num_train_steps == 7
    assert config.model_config.action_horizon == 3