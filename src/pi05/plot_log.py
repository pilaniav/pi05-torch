"""
Plot training curves from a train.log written by pi05.train.

Usage:
    uv run python -m pi05.plot_log --log-path train_results/pi05_b1k_4xa100/train.log
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files only, no window needed
import matplotlib.pyplot as plt  # noqa: E402
import tyro  # noqa: E402

# Matched anywhere in a line, so the logging prefix (time, level, name) is skipped.
TRAIN_LINE = re.compile(r"step (\d+)\s+loss ([\d.]+)\s+grad_norm ([\d.]+)\s+lr ([\d.e+-]+)")
VAL_LINE = re.compile(r"step (\d+)\s+val_loss ([\d.]+)")


def parse_log(path: Path) -> tuple[dict[int, tuple[float, float, float]], dict[int, float]]:
    """
    Read train.log into {step: (loss, grad_norm, lr)} and {step: val_loss}.

    A resumed run logs some steps twice; the later line wins, as it belongs to the run
    that continued.
    """
    train = {}
    val = {}
    for line in path.read_text().splitlines():
        match = TRAIN_LINE.search(line)
        if match:
            step = int(match.group(1))
            train[step] = (float(match.group(2)), float(match.group(3)), float(match.group(4)))
            continue
        match = VAL_LINE.search(line)
        if match:
            val[int(match.group(1))] = float(match.group(2))
    return train, val


def plot(log_path: Path, out: Path | None = None) -> None:
    """
    Save loss/val_loss, grad_norm and lr curves as a PNG (default: next to the log).
    """
    train, val = parse_log(log_path)
    if out is None:
        out = log_path.with_suffix(".png")

    train_steps = sorted(train)
    loss = []
    grad_norm = []
    lr = []
    for step in train_steps:
        loss.append(train[step][0])
        grad_norm.append(train[step][1])
        lr.append(train[step][2])

    val_steps = sorted(val)
    val_loss = []
    for step in val_steps:
        val_loss.append(val[step])

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(train_steps, loss, label="train loss (mean per log interval)")
    axes[0].plot(val_steps, val_loss, marker="o", label="val loss")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(train_steps, grad_norm)
    axes[1].set_ylabel("grad norm (before clipping)")

    axes[2].plot(train_steps, lr)
    axes[2].set_ylabel("learning rate")
    axes[2].set_xlabel("step")

    for ax in axes:
        ax.grid(True, alpha=0.3)
    fig.suptitle(log_path.parent.name)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    tyro.cli(plot)