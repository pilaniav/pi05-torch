from dataclasses import dataclass, field
from pathlib import Path

from pi05.models.pi05_config import Pi05Config


@dataclass(frozen=True)
class TrainConfig:
    exp_name: str
    model_config: Pi05Config = field(default_factory=Pi05Config)
    pretrained_path: Path | None = None  # None = random init
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
    dataset_dir: Path | None = None  # prepared_data/<name>; None = random fake batches
    tokenizer_path: Path = Path("pretrained/paligemma_tokenizer.model")
    num_workers: int = 4  # data loader worker processes per GPU
    val_interval: int = 1000  # steps between validation losses; 0 = off
    num_val_batches: int = 10  # fixed validation batches, built once at start
    overfit_one_batch: bool = False  # debug: train on the first batch only (loss should -> ~0)

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
    ),
    "dev": TrainConfig(  # tiny model on real data (prepared_data/dev), CPU
        exp_name="dev",
        model_config=Pi05Config(
            vision_variant="siglip_tiny_224",
            vlm_variant="gemma_tiny_full_vocab",
            expert_variant="gemma_tiny_expert",
        ),
        dataset_dir=Path("prepared_data/dev"),
        batch_size=4,
        num_train_steps=50,
        decay_steps=50,
        warmup_steps=2,  # short warmup and high LR: learning is visible within 50 steps
        peak_lr=1e-3,
        log_interval=5,
        save_interval=10,
        precision="float32",
        num_workers=2,
        val_interval=10,
        num_val_batches=2,
    ),
    "pi05_b1k": TrainConfig(  # full pi0.5, fine-tuned on the BEHAVIOR tasks in dataset_dir
        exp_name="pi05_b1k",
        model_config=Pi05Config(),  # full model: so400m + gemma_2b + gemma_300m
        pretrained_path=Path("pretrained/pi05_base/model.safetensors"),
        dataset_dir=Path("prepared_data/behavior"),
        batch_size=32,  # global: 4 per GPU on 8 GPUs
        num_train_steps=50_000,  # as b1k; check the val curve, it may plateau earlier
        decay_steps=50_000,
        warmup_steps=1_000,
        peak_lr=2.5e-5,
        precision="bfloat16",
        num_workers=4,  # per GPU
        save_interval=1_000,
        val_interval=1_000,
        num_val_batches=10,
        compile=True,  # measured on A100: 0.58 vs 0.80 s/step, ~4 min to compile at start
    ),
}


def get_train_config(name: str) -> TrainConfig:
    if name not in TRAIN_CONFIGS:
        raise ValueError(f"Unknown train config {name!r}. Options: {list(TRAIN_CONFIGS)}")
    return TRAIN_CONFIGS[name]