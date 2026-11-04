# Setup

What a fresh machine (laptop, Docker image, vast.ai) needs before anything runs.
All commands run from the repo root.

## 1. System packages

    sudo apt install ffmpeg      # FFmpeg 4-9; torchcodec needs it to read videos

## 2. Python environment

    uv sync                      # installs pyproject.toml deps, exact versions from uv.lock

- torchcodec >= 0.12 requires torch >= 2.11 (pinned in pyproject.toml).
- PyAV bundles its own FFmpeg; it only needs the H.264 encoder.

Check:

    python -c "import av; print('libx264' in av.codecs_available)"     # True
    python -c "from torchcodec.decoders import VideoDecoder; print('ok')"

## 3. Accounts

    huggingface-cli login        # higher download rate limits; needed if a dataset is gated

## 4. Folder layout

All gitignored, relative to the repo root:

    raw_data/behavior/       downloaded parquet, RGB videos, meta     <- pi05.data_pipeline.download
    prepared_data/<name>/    same layout: parquets, 224 videos,       <- pi05.data_pipeline.prepare_dataset
                             meta/episodes.jsonl (+ train/val split),
                             norm_stats.json, prepared_data_info.json
    pretrained/              sentencepiece tokenizer, pi05_base       <- manual, once (section 5)
    checkpoints/<exp_name>/  training runs: checkpoints, train.log    <- pi05.train

## 5. External files

Dataset (script):

    source: Hugging Face dataset behavior-1k/2025-challenge-demos
    to:     raw_data/behavior/
    how:    python -m pi05.data_pipeline.download ...

Sentencepiece tokenizer (PaliGemma vocabulary, manual, once):

    source: https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
    to:     pretrained/paligemma_tokenizer.model
    how:    curl -L -o pretrained/paligemma_tokenizer.model <source>

pi05_base weights (manual, once: download JAX, convert to PyTorch):

    source: gs://openpi-assets/checkpoints/pi05_base   (JAX/orbax only)
    to:     pretrained/pi05_base/model.safetensors
    how:    openpi's examples/convert_jax_model_to_pytorch.py, in a separate env
            (jax, orbax, flax, openpi). Pass --precision float32 (default is bf16).
            The checkpoint path must contain "pi05", or it silently converts as pi0.
    check:  load_pretrained must load it with no errors (strict).
    alt:    LeRobot's PyTorch pi0.5 on Hugging Face (verify names/values match).

Norm stats (script): computed by prepare into prepared_data/<name>/norm_stats.json.

## 6. Before a GPU run (vast.ai)

Machines:
- Training: A100/H100 80 GB. Evaluation / RL in Isaac Sim: RTX GPUs (RT cores).
- Prefer 1 node x 8 GPUs over 2 nodes unless there is a fast interconnect.

Smoke run on one GPU first:
- Real model at batch 32 / world_size: peak GPU memory, sec/step, compile on vs off.
- Overfit one batch: loss -> ~0 within a few hundred steps.
- Save + resume a checkpoint; check disk per checkpoint (~40 GB).

Habits:
- Start training inside tmux, so it survives a dropped SSH connection.
- Sync code with git (push from laptop, pull on server); commit fixes made on the server.
- Copy checkpoints off the instance before destroying it.

## 7. Quick start

The full pipeline, in order:

    # 1. Download raw data (resumable; re-run after rate limits)
    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio picking_up_trash

    # 2. Prepare: re-encode videos at 224, copy parquets, train/val split, norm stats
    uv run python -m pi05.data_pipeline.prepare_dataset

    # 3. Train
    uv run python -m pi05.train debug          # TODO: real config once the data loader exists

Small local test (a few episodes, separate folder):

    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio --episodes-per-task 10
    uv run python -m pi05.data_pipeline.prepare_dataset --out-dir prepared_data/dev --val-episodes-per-task 2

Recompute only the norm stats (e.g. after deleting raw data):

    uv run python -m pi05.data_pipeline.norm_stats --dataset-dir prepared_data/behavior

Each script's docstring lists more options; `--help` shows all flags.
