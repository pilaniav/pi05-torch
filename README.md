# pi05-torch

A from-scratch PyTorch reimplementation of [π0.5](https://www.physicalintelligence.company/blog/pi05),
the vision-language-action model from Physical Intelligence, together with its training and
data pipeline. The target setting is fine-tuning on tasks from the
[BEHAVIOR-1K](https://behavior.stanford.edu/) benchmark.

Built as a learning and research project, with readability as a priority: small modules,
explicit code, and comments that explain the reasoning behind each design choice.

## Getting started

See [docs/setup.md](docs/setup.md) for setup, data preparation and training.

## License

Copyright 2026 Vinay Pilania

Licensed under the Apache License, Version 2.0; see [LICENSE](LICENSE).

## Acknowledgments

This project was written while studying [openpi](https://github.com/Physical-Intelligence/openpi)
(Apache-2.0) by Physical Intelligence, and the BEHAVIOR-1K baselines. The model architecture,
prompt format and data transforms follow openpi so that its pretrained weights can be used.

Pretrained weights (`lerobot/pi05_base`, from [LeRobot](https://github.com/huggingface/lerobot))
and the PaliGemma tokenizer are not part of this repository; they are distributed separately
under the Gemma license.