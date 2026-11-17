# pi05-torch

A from-scratch PyTorch reimplementation of [π0.5](https://www.physicalintelligence.company/blog/pi05),
the vision-language-action model from Physical Intelligence, together with its training and
data pipeline. The target setting is fine-tuning on tasks from the
[BEHAVIOR-1K](https://behavior.stanford.edu/) benchmark.

Built as a learning and research project, with readability as a priority: small modules,
explicit code, and comments that explain the reasoning behind each design choice.

The model loads the public pi0.5 base weights (`lerobot/pi05_base`) exactly, and tiny
configurations let the whole pipeline, from data to training, run on a CPU laptop.

## Features

**Model**
- SigLIP so400m vision encoder, Gemma 2B language model and a Gemma 300M action expert
  with adaptive RMSNorm and gated residuals, conditioned on the timestep
- Joint attention between the vision-language prefix and the action expert
- Flow-matching training loss; 10-step (default) Euler sampling with a KV-cached prefix
- Robot state as discrete tokens in the prompt, as in pi0.5
- Strict weight loading: openpi/LeRobot checkpoints (all 812 tensors checked) or our own

**Data pipeline (BEHAVIOR-1K)**
- Parallel, resumable download from Hugging Face with retries and backoff
- Videos re-encoded to 224x224 H.264 (keyframe every 10 frames, no B-frames) for fast,
  frame-exact random access
- 23-dim state and action extracted from BEHAVIOR's 256-dim robot state
- Exact quantile normalization statistics over the train split
- openpi's prompt format and image augmentation (crop, rotation, color), training only

**Training**
- Multi-GPU with DistributedDataParallel (`torchrun`)
- Memory: bf16 autocast with fp32 weights, gradient checkpointing, gradients kept in
  DDP's buffers (saves ~13 GB), and parameters that get no gradient frozen.
  Full fine-tuning fits 4 samples per 80 GB GPU (~66 GB peak)
- Speed: `torch.compile` (0.80 -> 0.58 s/step on an A100), limited codec threads in data prep
- AdamW with warmup and cosine decay, gradient clipping
- Atomic checkpoints, resume, pruning of old checkpoints; validation loss on fixed batches
- Training curves plotted from the log

**Testing**
- Unit tests that run on CPU with tiny model sizes

## Results (preliminary, ongoing)

First-hand results from our first full fine-tuning run, as a reference. Not yet
evaluated in simulation; no checkpoint is released.

| Item              | Value                                   |
|-------------------|-----------------------------------------|
| Tasks             | turning_on_radio, picking_up_trash      |
| Episodes          | 400 (380 train / 20 val)                |
| Train frames      | 1.42M                                   |
| Start weights     | lerobot/pi05_base (full fine-tuning)    |
| Hardware          | 4x A100-SXM4-80GB (NVLink), DDP         |
| Steps             | 50k, global batch 16 (~0.56 epoch)      |
| Learning rate     | peak 2.5e-5, cosine decay to 2.5e-6     |
| Speed             | 0.66 s/step                             |
| Peak GPU memory   | 65.4 GB per GPU                         |
| Time              | ~9h40m                                  |
| Cost              | ~$52 (rented)                           |
| Train loss        | 0.18 (step 100) -> 0.0065 (step 50k)    |
| Validation loss   | 0.030 (step 1k) -> 0.017 (step 50k)     |

Validation loss improved mostly in the first ~4k steps and stayed nearly flat
afterwards, while training loss kept falling. Task success in simulation is the
real measure and will be reported once evaluation is in place.

## Getting started

See [docs/setup.md](docs/setup.md) for setup, data preparation and training.

## Roadmap

The next steps are in progress; the extensions will follow as time allows.

**Next**
1. Policy server: serve the trained model over openpi's websocket protocol
2. Evaluation: run BEHAVIOR-1K's evaluation in simulation through the policy server,
   using a ready-to-use Docker image (BEHAVIOR-1K 2025 code, Isaac Sim, WebRTC
   viewing) for any machine with an NVIDIA RTX GPU, local or rented
3. RL fine-tuning: one algorithm in depth, with new reward modeling that rewards task
   success while penalizing unnecessary base and arm motion

**Possible extensions**
- Gradient accumulation: a larger global batch on fewer GPUs
- FSDP: full fine-tuning on GPUs with less than 80 GB
- LoRA: fine-tuning with much less GPU memory
- Periodic checkpoints kept from pruning (weights only), to compare several in simulation
- Profiling GPU utilization. Observed: GPU utilization alternates between 100% and 0%
  during training (4 x A100, DDP).
- Profiling a training step: profile a few steps with torch.profiler to see where each
  step's time goes (forward, backward, gradient all-reduce, optimizer, data loading).
- Reusing a run's norm stats when training starts from its weights
- Scaling to the BEHAVIOR 2026 challenge data (100 tasks, 20,000 demos, ~1,950 h;
  LeRobot v3 format, so download and preparation need adapting)

## License

Copyright 2026 Vinay Pilania

Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).

## Acknowledgments

This project was written while studying [openpi](https://github.com/Physical-Intelligence/openpi)
(Apache-2.0) by Physical Intelligence, and the
[BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) baselines. The model architecture,
prompt format and data transforms follow openpi so that its pretrained weights can be used.

Pretrained weights (`lerobot/pi05_base`, from [LeRobot](https://github.com/huggingface/lerobot))
and the PaliGemma tokenizer are not part of this repository; they are distributed separately
under the Gemma license. Training data comes from the BEHAVIOR-1K 2025 challenge demos
(`behavior-1k/2025-challenge-demos` on Hugging Face), also not part of this repository.