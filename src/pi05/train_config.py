from dataclasses import dataclass, field
from pathlib import Path

from pi05.models.pi05_config import Pi05Config


@dataclass(frozen=True)
class TrainConfig:
    exp_name: str
    model_config: Pi05Config = field(default_factory=Pi05Config)
    pretrained_path: str | None = None
    checkpoint_base_dir: Path = Path("checkpoints")  # parent of all runs, relative to launch dir

    # Optimization
    batch_size: int = 32  # global, split across GPUs
    num_train_steps: int = 30_000
    peak_lr: float = 2.5e-5  # LR at end of warmup
    end_lr: float = 2.5e-6  # LR after cosine decay
    warmup_steps: int = 1_000
    decay_steps: int = 30_000  # counted from step 0, includes warmup
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
    clip_grad_norm: float = 1.0

    # Runtime
    seed: int = 42
    precision: str = "bfloat16"  # autocast dtype for forward: "bfloat16" or "float32"
    gradient_checkpointing: bool = True
    log_interval: int = 100
    save_interval: int = 1_000
    resume: bool = False  # continue from latest checkpoint in checkpoint_dir
    keep_last_checkpoints: int = 2  # older checkpoints are deleted
    compile: bool = False  # torch.compile the model

    @property
    def checkpoint_dir(self) -> Path:
        return (self.checkpoint_base_dir / self.exp_name).resolve()


TRAIN_CONFIGS = {
    "debug": TrainConfig(
        exp_name="debug",
        model_config=Pi05Config(
            vision_variant="siglip_tiny",
            vlm_variant="gemma_tiny",
            expert_variant="gemma_tiny_expert",
        ),
        num_train_steps=20,
        decay_steps=20,
        log_interval=5,
        save_interval=10,
        precision="float32",
    )
}


def get_train_config(name: str) -> TrainConfig:
    if name not in TRAIN_CONFIGS:
        raise ValueError(f"Unknown train config {name!r}. Options: {list(TRAIN_CONFIGS)}")
    return TRAIN_CONFIGS[name]