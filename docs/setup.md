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

## 4. Serving and evaluation

### 4.1 Model input and output

What the model sees and returns at inference (robot: BEHAVIOR's R1Pro). Training uses
the same processing, plus augmentation.

**Inputs**

| Input   | Format                                                                        |
|---------|-------------------------------------------------------------------------------|
| Images  | 3 cameras, in this order: head, left wrist, right wrist; 224x224 RGB uint8   |
|         | (the eval sends RGBA; alpha is dropped), scaled to [-1, 1]                    |
| State   | 23 dims taken from BEHAVIOR's 256-dim proprioception (order below)            |
| Prompt  | the task's text, e.g. "Turn on the radio receiver that's on the table in     |
|         | the living room."                                                             |

State order (23 dims):

    base velocity 3 | torso 4 | arm L 7 | arm R 7 | gripper L 1 | gripper R 1

- Gripper value = opening width, the sum of its two finger joints.
- The state is normalized with the training run's norm stats (q01 -> -1, q99 -> +1),
  split into 256 bins, and written into the prompt (no separate state input):

      Task: {task text}, State: {bins};\nAction:

  tokenized with the PaliGemma tokenizer (at most 200 tokens).

**Output**

An action chunk of 50 steps x 23 dims (the model's 32 dims, minus padding),
unnormalized with the training run's norm stats: absolute joint targets, with the base
as a velocity and each gripper as a continuous opening width.

Action order (23 dims, index ranges):

    base 0-2 | torso 3-6 | arm L 7-13 | gripper L 14 | arm R 15-21 | gripper R 22

- Same order as the dataset's action column and BEHAVIOR's R1Pro action space.
- It differs from the state order: in the state, both arms come before both grippers.
- One chunk covers 50 steps at 30 Hz (~1.7 s); the server returns one action per
  eval step (4.2).

### 4.2 Policy server

Serves a trained checkpoint to BEHAVIOR's eval over a websocket, using the eval's own
protocol. It runs in this repo's environment; the eval runs in BEHAVIOR's (4.3).

    uv run python -m pi05.serving.serve_policy \
        --checkpoint-dir train_results/<exp-name>/<step> \
        --tasks-file train_results/<exp-name>/data/meta/tasks.jsonl \
        --task-name turning_on_radio

- `--checkpoint-dir`: model.safetensors + the training run's norm_stats.json.
- `--tasks-file`: maps the task name to the exact prompt used in training.
- One server serves one task: the eval sends no prompt, so it is fixed at startup.

| Flag            | Default             | Meaning                                         |
|-----------------|---------------------|-------------------------------------------------|
| `--port`        | 8000                | the eval connects here                          |
| `--mode`        | `temporal_ensemble` | model called every step; the step's actions from the last 10 chunks are averaged (grippers from the newest) |
|                 | `receding_horizon`  | model called every `--replan-steps` (10) steps; ~10x fewer calls |
| `--save-obs`    | off                 | save the first observation received to `--results-dir` (default eval_results/) |
| `--device`      | cuda if available   | weights in bf16 on GPU, float32 on CPU          |

Protocol, as BEHAVIOR's client expects it: `GET /healthz` answers 200; on connect the
server sends metadata ({}); `{"reset": True}` resets the chunk history without a reply;
each observation gets `{"action": (1, 23)}` back; messages are msgpack with numpy arrays.

Log lines to expect:

    Loaded <checkpoint> on cuda (torch.bfloat16)
    Serving on ws://0.0.0.0:8000
    Client connected from (...)
    infer_ms over the last 100 actions: mean 225.3, max 238.2

Measured on an RTX 4090: ~7.4 GB of GPU memory, ~225 ms per model call (`temporal_ensemble`).
Stop with Ctrl-C.

To check a checkpoint without a GPU, run the same command with `CUDA_VISIBLE_DEVICES=`:
it loads in about a minute on CPU (needs ~14 GB of RAM) and serves, slowly.

### 4.3 BEHAVIOR eval

BEHAVIOR's own eval script drives the policy server: it runs the simulator, sends
observations, executes the returned actions and writes metrics. Nothing in BEHAVIOR's
code is changed.

**Machine**

- An NVIDIA **RTX** GPU: the simulator (Isaac Sim) needs RT cores; A100/H100 can't run it.
- 2x RTX 4090 (24 GB): one for the server (~7.4 GB), one for the simulator (~13.8 GB).
  Both on one 24 GB card do not fit reliably.
- Driver **580.x (CUDA 13.0)**. Newer drivers (600+) crash Isaac Sim 4.5's renderer;
  older ones (570.x, CUDA 12.8) are too old for this repo's torch build.

**Software**

- BEHAVIOR-1K (2025 challenge version, Isaac Sim 4.5) in its own environment
  (e.g. conda `behavior`), with its data: `./setup.sh --dataset` in BEHAVIOR-1K.
- This repo, set up as in section 1, with the tokenizer (2.1).
- The checkpoint. Uploading 13 GB from home is slow; push it once to a private
  Hugging Face repo (from a rented machine it takes minutes), then download it anywhere:

      uv run hf upload <hf-user>/<repo> train_results/<exp-name>/<step> . --private
      uv run hf download <hf-user>/<repo> --local-dir train_results/<exp-name>/<step>

**Run (two terminals)**

Terminal 1, this repo, server on GPU 0:

    RUN=$PWD/eval_results/<run-name> && mkdir -p $RUN
    CUDA_VISIBLE_DEVICES=0 uv run python -m pi05.serving.serve_policy \
        --checkpoint-dir train_results/<exp-name>/<step> \
        --tasks-file train_results/<exp-name>/data/meta/tasks.jsonl \
        --task-name turning_on_radio 2>&1 | tee $RUN/server.log

Terminal 2, BEHAVIOR-1K root and environment, simulator on GPU 1, once the server
prints "Serving on ...":

    RUN=/absolute/path/to/pi05-torch/eval_results/<run-name>
    unset CUDA_VISIBLE_DEVICES
    export OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=1
    python OmniGibson/omnigibson/learning/eval.py policy=websocket \
        task.name=turning_on_radio log_path=$RUN 2>&1 | tee $RUN/eval.log

- Select the simulator's GPU with `OMNIGIBSON_GPU_ID`, never `CUDA_VISIBLE_DEVICES`:
  Isaac Sim renders through Vulkan, and the mismatch with CUDA's device numbering
  crashes it.
- `log_path` must be absolute: eval.py runs from the BEHAVIOR-1K folder.
- If the environment sets `OMNIGIBSON_REMOTE_STREAMING`, the simulator streams to
  http://localhost:8211 (forward that port).

| Eval option                    | Meaning                                                 |
|--------------------------------|---------------------------------------------------------|
| `task.name=...`                | the task (must match the server's `--task-name`)        |
| `'eval_instance_ids=[0,1]'`    | a subset of the 10 test instances (default: all)        |
| `eval_on_train_instances=true` | use the demos' starting states instead                  |
| `model.port=8001`              | the server's port (default 8000)                        |
| `write_video=false`            | skip videos                                             |
| `max_steps=...`                | time limit (default: 2x the mean human demo length)     |

**Task instances**

Each task has 10 official test instances (positions 0-9 of the 20 listed per task in
BEHAVIOR's `test_instances.csv`). Their starting states are not among the training
demos. One rollout each; an episode ends on success or at the time limit
(e.g. 4300 steps, ~2.4 min of sim time, for turning_on_radio).

**Outputs** (in `$RUN`)

    metrics/<task>_<instance>_0.json   q_score (final: 1.0 = success, else partial
                                       credit), time, base and arm distances
    videos/<task>_<instance>_0.mp4     head and wrist cameras
    server.log, eval.log

Aggregate results from the json files: the success rate printed at the end of eval.log
covers only the last episode.

**Timing** (2x RTX 4090, `temporal_ensemble`): ~0.30 s per simulator step (~225 ms of
it is the model), so ~21 min for a turning_on_radio instance that runs to the time limit.

### 4.4 Several tasks

Each eval run covers one task, and each server serves one task, so several tasks
means one (server, eval) pair per task. BEHAVIOR ships no multi-task runner.

**One after the other** (one pair of GPUs): for each task, start the server with
`--task-name <task>`, run the eval with `task.name=<task>`, then stop the server.
Use the same task name on both sides: a mismatch is not detected, the model just
gets the wrong prompt.

**In parallel**: one pair per task (or per subset of instances), each with its own port,
and never two evals on one server (the chunk history is per server):

    serve_policy ... --task-name turning_on_radio --port 8000
    serve_policy ... --task-name picking_up_trash --port 8001

    eval.py ... task.name=turning_on_radio model.port=8000 log_path=$RUN
    eval.py ... task.name=picking_up_trash model.port=8001 log_path=$RUN

Runs can share one `log_path`: file names include the task and instance.

**Loop pattern** (a sketch, not a tested script):

    for TASK in turning_on_radio picking_up_trash; do
        uv run python -m pi05.serving.serve_policy ... --task-name $TASK --port 8000 &
        SERVER_PID=$!
        until curl -sf localhost:8000/healthz; do sleep 5; done   # wait for the model
        <BEHAVIOR python> OmniGibson/omnigibson/learning/eval.py policy=websocket \
            task.name=$TASK log_path=$RUN
        kill $SERVER_PID
    done

Restarting the server per task costs about a minute (model loading), small next to the
eval itself.

## 5. Folder layout and key files

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
      serving/                   trained model -> BEHAVIOR eval
      * serve_policy.py            policy server entry point: CLI, task prompt, starts the server
        policy.py                  Policy (inputs -> action chunk) and B1KPolicy (BEHAVIOR keys,
                                   temporal ensemble or receding horizon, one action per call)
        websocket_server.py        BEHAVIOR's websocket protocol (/healthz, reset, act), msgpack
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
    eval_results/<run-name>/     eval metrics (json), videos, server and  <- eval (4.3)
                                 eval logs, saved observation