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


## 4. Folder layout and key files

Relative to the repo root.

### Code (in git)

    src/pi05/
      models/                the network: Gemma, SigLIP, pi0.5 (pi05.py)
      data_pipeline/         raw data -> training batches
        download.py            download parquet + RGB videos
        prepare_dataset.py     224 videos, train/val split, norm stats
        dataset.py             dataset + data loader (+ transforms.py, norm_stats.py)
      train.py               training entry point (+ train_config.py, checkpoint.py)
    tests/unit/              unit tests
    docs/setup.md            this file
    pyproject.toml, uv.lock  dependencies (exact versions in uv.lock)
    CLAUDE.md                instructions for Claude Code

### Data and outputs (gitignored)

    raw_data/behavior/       downloaded parquet, RGB videos, meta       <- pi05.data_pipeline.download
    prepared_data/<name>/    224 videos, parquets, train/val split,     <- pi05.data_pipeline.prepare_dataset
                             norm_stats.json
    pretrained/              sentencepiece tokenizer, pi05_base weights <- manual, once (section 5)
    checkpoints/<exp_name>/  checkpoints (+ norm_stats.json), train.log <- pi05.train


## 5. External files

Dataset (script):

    source: Hugging Face dataset behavior-1k/2025-challenge-demos
    to:     raw_data/behavior/
    how:    uv run python -m pi05.data_pipeline.download ...

Sentencepiece tokenizer (PaliGemma vocabulary, manual, once):

    source: https://storage.googleapis.com/big_vision/paligemma_tokenizer.model
    to:     pretrained/paligemma_tokenizer.model
    how:    mkdir -p pretrained
            curl -L -o pretrained/paligemma_tokenizer.model <source>
    used:   data loader and policy server (TrainConfig.tokenizer_path)

pi05_base weights (PyTorch, from LeRobot, once):

    source: Hugging Face model lerobot/pi05_base   (float32 safetensors, Gemma license)
            Same key names as openpi's PyTorch path (verified: all 812 tensors).
    to:     pretrained/pi05_base/model.safetensors
    how:    huggingface-cli download lerobot/pi05_base --include "*.safetensors" \
                --local-dir pretrained/pi05_base
    used:   training start (TrainConfig.pretrained_path), loaded by load_pretrained (strict)
    alt:    openpi's JAX checkpoint gs://openpi-assets/checkpoints/pi05_base, converted with
            openpi's examples/convert_jax_model_to_pytorch.py in a separate env (jax, orbax,
            flax, openpi). Pass --precision float32; the path must contain "pi05".

Norm stats (script, no download):

    made:   by prepare_dataset into prepared_data/<name>/norm_stats.json
    redo:   uv run python -m pi05.data_pipeline.norm_stats --dataset-dir prepared_data/<name>
    used:   data loader; copied into every checkpoint for the policy server


## 6. Before a GPU run (vast.ai)

Machines:
- Training: A100/H100 80 GB. Evaluation / RL in Isaac Sim: RTX GPUs (RT cores).
- Prefer 1 node x 8 GPUs over 2 nodes unless there is a fast interconnect.
- Enough CPU cores for the data loaders: at least (number of GPUs) x num_workers,
  e.g. 8 GPUs x 4 workers = 32 cores.

Set up the instance:
1. System and Python packages, Hugging Face login: sections 1-3.
2. Tokenizer and pi05_base weights: section 5.
3. Download and prepare both tasks on the instance itself (fast network, many cores):

       uv run python -m pi05.data_pipeline.download
       uv run python -m pi05.data_pipeline.prepare_dataset --num-workers 32
4. Check pi05_b1k in train_config.py: pretrained_path and dataset_dir must match
   the instance's paths (or override them: --pretrained-path ... --dataset-dir ...).

Smoke run on one GPU first:
- load_pretrained loads the weights with no errors.
- Real model at batch 32 / world_size: peak GPU memory, sec/step, compile on vs off.
- Overfit one batch: loss -> ~0 within a few hundred steps.
- Save + resume a checkpoint; check disk per checkpoint (~40 GB).

Habits:
- Start training inside tmux, so it survives a dropped SSH connection.
- Sync code with git (push from laptop, pull on server); commit fixes made on the server.
- Copy checkpoints off the instance before destroying it.


## 7. Quick start

Small local test on a laptop (CPU, a few minutes):

    # 1. Download 10 episodes of one task
    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio --episodes-per-task 10

    # 2. Prepare them into a separate folder (2 validation episodes)
    uv run python -m pi05.data_pipeline.prepare_dataset --out-dir prepared_data/dev --val-episodes-per-task 2

    # 3. Tokenizer, once (section 5)
    mkdir -p pretrained
    curl -L -o pretrained/paligemma_tokenizer.model https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

    # 4. Train the tiny model on it (50 steps; loss and val_loss should drop)
    uv run python -m pi05.train dev

Check the training code alone, without any data (random batches):

    uv run python -m pi05.train debug

Full run (GPU, see section 6):

    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio picking_up_trash
    uv run python -m pi05.data_pipeline.prepare_dataset
    uv run python -m pi05.train pi05_b1k     # full model; trains on all tasks in dataset_dir

Recompute only the norm stats (e.g. after deleting raw data):

    uv run python -m pi05.data_pipeline.norm_stats --dataset-dir prepared_data/behavior

A run refuses to start over existing checkpoints: delete checkpoints/<exp_name>/,
or continue it with --resume True. Each script's docstring lists more options;
--help shows all flags.