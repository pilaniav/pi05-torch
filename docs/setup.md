# Setup and usage

How to install pi05-torch, get the external files, and run training.
After cloning, all commands run from the repo root.

The code runs on CPU with tiny model configs: `debug` (random data) and `dev`
(a few real episodes), so everything can be tested on a laptop. Full training
needs GPUs with 80 GB or more (A100/H100 80 GB, H200, B200).

## 1. Install

### 1.1 System packages

    apt install git curl ffmpeg      # ffmpeg 4-9: torchcodec decodes videos with it
    apt install tmux                 # GPU instances: keeps long jobs alive if SSH drops

### 1.2 Repository

    git clone https://github.com/pilaniav/pi05-torch.git
    cd pi05-torch

### 1.3 Python environment

    curl -LsSf https://astral.sh/uv/install.sh | sh    # uv, once per machine
    uv sync                                            # exact versions from uv.lock

Images with their own Python environment (e.g. vast.ai's /venv/main): uv warns that
VIRTUAL_ENV points elsewhere. Ignore it, or use uv's own Python:

    unset VIRTUAL_ENV
    UV_PYTHON_PREFERENCE=only-managed uv sync --python 3.12

Check:

    uv run python -c "import av; print('libx264' in av.codecs_available)"        # True
    uv run python -c "from torchcodec.decoders import VideoDecoder; print('ok')"

### 1.4 Hugging Face login

    uv run hf auth login       # read token from huggingface.co/settings/tokens

Higher download rate limits, and needed for gated datasets or models.
(Older huggingface_hub versions: `huggingface-cli login`.)

### 1.5 GPU checks (any machine with an NVIDIA GPU)

    nvidia-smi              # GPUs, memory, driver "CUDA Version" (>= 13.0)
    nvidia-smi topo -m      # multi-GPU: NV# = NVLink between GPUs; PIX/PHB/SYS = PCIe only
    df -h .                 # free space on the repo's disk, where data and checkpoints go
    uv run python -c "import torch; print(torch.cuda.device_count())"    # GPUs torch can use

- Disk for full training: ~37 GB per checkpoint; up to 3 exist at once (2 kept +
  1 being written), plus weights and data.
- In a container, `nproc` and `free` show the whole host, not your rented share:
  take cores and RAM from the rental listing (sets prepare_dataset's `--num-workers`, 2.3).

### 1.6 CPU testing (tiny configs)

For light testing on CPU: unit tests, `debug` and `dev`.

On a machine with no NVIDIA GPU nothing is needed: torch uses the CPU.

With a GPU that torch cannot use (e.g. a broken driver, CUDA error 804), or to test
on CPU on purpose, hide the GPU:

    CUDA_VISIBLE_DEVICES= uv run python -m pi05.train dev     # one command
    echo 'export CUDA_VISIBLE_DEVICES=' >> ~/.bashrc          # every new terminal

## 2. External files

| File                 | Needed by                     | Location                                   |
|----------------------|-------------------------------|--------------------------------------------|
| Tokenizer            | `dev`, `pi05_b1k`, inference  | pretrained/paligemma_tokenizer.model       |
| pi05_base weights    | `pi05_b1k`                    | pretrained/pi05_base/model.safetensors     |
| Dataset              | `dev`, `pi05_b1k`             | raw_data/behavior -> prepared_data/<name>  |
| Norm stats           | `dev`, `pi05_b1k`, inference  | prepared_data/<name>/norm_stats.json       |

`debug` needs none of them (random data, random weights).

### 2.1 Tokenizer

PaliGemma's sentencepiece tokenizer: turns the prompt text into token IDs
(the token embeddings are part of the model weights).

    mkdir -p pretrained
    curl -L -o pretrained/paligemma_tokenizer.model \
        https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

### 2.2 pi05_base weights

Pretrained pi0.5 from LeRobot (PyTorch, float32 safetensors, Gemma license),
the starting point for fine-tuning:

    uv run hf download lerobot/pi05_base --include "*.safetensors" \
        --local-dir pretrained/pi05_base

- Same key names as openpi's PyTorch model; `load_pretrained` maps them to ours
  and checks every tensor (strict).
- `--pretrained-path` also accepts our own checkpoint's model.safetensors, to start
  a new run from trained weights.
- Alternative: openpi's JAX checkpoint (gs://openpi-assets/checkpoints/pi05_base),
  converted with openpi's examples/convert_jax_model_to_pytorch.py in a separate
  environment (`--precision float32`; the output path must contain "pi05").

### 2.3 Dataset

BEHAVIOR-1K 2025 challenge demos (Hugging Face: behavior-1k/2025-challenge-demos):
50 household tasks, 10,000 teleoperated demonstrations (about 200 per task).

Download parquet files and RGB videos into raw_data/behavior/:

    # two tasks, all episodes
    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio picking_up_trash

    # one task, 10 episodes (small test set)
    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio --episodes-per-task 10

    # all 50 tasks, 10 episodes each
    uv run python -m pi05.data_pipeline.download --all-tasks --episodes-per-task 10

- Without `--episodes-per-task`, all episodes of each task are downloaded.
- Videos: RGB only (depth and segmentation are not used), from all three cameras
  (head, left wrist, right wrist). `--cameras` can select fewer, but training
  expects all three.
- If the download stops (e.g. Hugging Face rate limit), run the same command again:
  files already on disk are skipped.
- More options (workers, retries): see the docstring in
  src/pi05/data_pipeline/download.py, or `--help`.

Prepare: re-encode videos to 224x224, split train/val, compute norm stats:

    uv run python -m pi05.data_pipeline.prepare_dataset --num-workers 32

- Output: prepared_data/behavior/ (the `pi05_b1k` default dataset_dir).
- `--num-workers`: about the number of CPU cores you rented.
- Re-runs skip finished episodes; `--delete-raw` frees raw_data/ as it goes.

Although prepare_dataset writes the norm stats, they can be recomputed separately
(e.g. after changing the train/val split):

    uv run python -m pi05.data_pipeline.norm_stats --dataset-dir prepared_data/behavior

- They are copied into every checkpoint: inference must use the training run's stats.

## 3. Quick start

### 3.1 CPU, random data (optional)

For exploring or changing the code: 3.1 and 3.2 run a tiny model on CPU in minutes.
To train the real model, skip to 3.3.

    uv run pytest tests/unit -q          # unit tests
    uv run python -m pi05.train debug    # tiny model, random batches, no files needed

### 3.2 CPU, small real data (optional)

Tiny model on 10 real episodes; needs the tokenizer (2.1).

    uv run python -m pi05.data_pipeline.download --tasks turning_on_radio --episodes-per-task 10
    uv run python -m pi05.data_pipeline.prepare_dataset --out-dir prepared_data/dev --val-episodes-per-task 2
    uv run python -m pi05.train dev      # 50 steps; loss and val_loss should drop

### 3.3 Full run (GPU)

Needs the tokenizer, pi05_base weights and the prepared dataset (section 2).
`pi05_b1k` reads them from their default paths; override with `--pretrained-path`
and `--dataset-dir`.

Smoke test on one GPU first (a few minutes):

    uv run python -m pi05.train pi05_b1k --exp-name smoke --batch-size 4 --num-train-steps 200

- No errors loading the weights; loss falls.
- Log line shows `mem` (peak GPU memory, ~66 GB at 4 per GPU) and s/step.
- Delete the smoke run afterwards: `rm -rf checkpoints/smoke`

Then train on all GPUs, inside tmux:

    tmux new -s train       # skip if already inside tmux (some images, e.g. vast.ai, start SSH in tmux)
    uv run torchrun --nproc-per-node 4 -m pi05.train pi05_b1k --exp-name pi05_b1k_4gpu --batch-size 16

- Inside tmux already? `echo $TMUX` prints a path. Open a new window with Ctrl-b c.
- `--batch-size` is global: per-GPU batch = batch size / GPUs, so it must divide
  evenly (4 per GPU fits 80 GB).
- Logs and checkpoints go to checkpoints/<exp-name>/ (train.log, one folder per
  saved step, last 2 kept).
- Detach with Ctrl-b d; reattach with `tmux attach -t train`.

Follow the run:

    tail -f checkpoints/<exp-name>/train.log          # all log lines, live
    grep val_loss checkpoints/<exp-name>/train.log    # validation loss only

### 3.4 Resume and other flags

    --resume                # continue from the latest checkpoint of --exp-name
    --compile               # torch.compile: ~25% faster steps, ~4 min to compile
    --no-compile            # turn it off (pi05_b1k has it on)
    --overfit-one-batch     # train on one batch repeatedly: loss should go near 0
    --help                  # all flags

- A run refuses to start over existing checkpoints: pass `--resume`, or use a new
  `--exp-name`.
- Boolean flags take no value: `--resume`, not `--resume True`.

### 3.5 Plot the log

    uv run python -m pi05.plot_log --log-path checkpoints/<exp-name>/train.log

Writes train.png next to the log: train/val loss, grad norm, learning rate.

### 3.6 After a run (rented instance)

Replace `<port>` and `<host>` with the instance's SSH details (e.g. vast.ai shows
`ssh -p 41234 root@203.0.113.7`), `/workspace/pi05-torch` with where you cloned
the repo on the instance, and `<step>` with the checkpoint folder to copy
(e.g. 50000, the last step).

On your own machine, copy the final checkpoint (without the optimizer state, which is
only needed to resume this exact run, ~25 GB) and the log:

    mkdir -p train_results/<exp-name>/<step>
    
    rsync -avP -e "ssh -p <port>" root@<host>:/workspace/pi05-torch/checkpoints/<exp-name>/<step>/{model.safetensors,norm_stats.json,metadata.json} train_results/<exp-name>/<step>/
    rsync -avP -e "ssh -p <port>" root@<host>:/workspace/pi05-torch/checkpoints/<exp-name>/train.log train_results/<exp-name>/

Check the weights arrived intact; both hashes must match:

    # on the instance
    sha256sum checkpoints/<exp-name>/<step>/model.safetensors

    # on your machine
    sha256sum train_results/<exp-name>/<step>/model.safetensors

Then destroy the instance (stopping it still bills for storage).

## 4. Folder layout and key files

Relative to the repo root. Files marked `*` are entry points: the scripts you run
(`uv run python -m pi05.<module>`). Start reading the code from them; everything
else is called from there.

### Code (in git)

    src/pi05/
      models/                    the network
        gemma.py                   Gemma decoder (2B VLM backbone, 300M action expert with adaRMS)
        siglip.py                  SigLIP vision encoder (so400m)
        paligemma.py               SigLIP + projector + Gemma
        paligemma_with_expert.py   VLM and action expert with joint attention
        pi05.py                    pi0.5: flow-matching loss, KV-cached action sampling
        gemma_config.py            Gemma sizes: 2B, 300M expert, tiny test sizes
        siglip_config.py           SigLIP sizes: so400m, tiny test sizes
        pi05_config.py             Pi05Config: picks the sizes (default: full model)
        observation.py             model input: images, image masks, prompt ids and mask
                                   (the state is tokenized inside the prompt)
      data_pipeline/             raw data -> training batches
      * download.py                download parquet + RGB videos from Hugging Face
      * prepare_dataset.py         224 videos, train/val split, norm stats
        b1k_state_action.py        23-dim state/action from BEHAVIOR's 256-dim state
      * norm_stats.py              quantile statistics over the train split
        transforms.py              normalization, prompt tokenizer, image augmentation
        dataset.py                 dataset, data loader, validation batches
    * train.py                   training entry point (single GPU or torchrun)
      train_config.py            training configs: debug, dev, pi05_b1k
      checkpoint.py              save/resume checkpoints, load pretrained weights
      distributed.py             process group setup for DDP
      log_setup.py               logging (rank 0 to console and train.log)
      fake_data.py               random batches for the debug config
    * plot_log.py                plot train.log curves
    tests/unit/                  unit tests
    docs/setup.md                this file
    pyproject.toml, uv.lock      dependencies (exact versions in uv.lock)
    README.md, LICENSE           overview, Apache-2.0 license

### Data and outputs (gitignored)

    raw_data/behavior/           downloaded parquet, RGB videos, meta     <- download (2.3)
    prepared_data/<name>/        224 videos, parquet, train/val split,    <- prepare_dataset (2.3)
                                 norm_stats.json
    pretrained/                  tokenizer, pi05_base weights             <- manual, once (2.1, 2.2)
    checkpoints/<exp_name>/      train.log, one folder per saved step     <- train (3.3)
      <step>/                    model.safetensors, optimizer.pt,
                                 metadata.json, norm_stats.json
    train_results/<exp_name>/    results copied off a rented instance     <- rsync (3.6)