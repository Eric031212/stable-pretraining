# Single-GPU SSL-EY CIFAR-10 recipe

This benchmark uses stable-pretraining `Module`, `DataModule` and `Manager`, with torchvision CIFAR-10 wrapped by `FromTorchDataset` and `MultiViewTransform`. It targets one CUDA GPU and the full 50000/10000 train/test splits. Do not mix its checkpoints with other architectures or recipes.

## Source and configurations

The reference is the author's `ssley` branch of [solo-learn at commit 11eac929a3cfb8fe28cefa573ef87e08fcf1eb27](https://github.com/jameschapman19/solo-learn/tree/11eac929a3cfb8fe28cefa573ef87e08fcf1eb27), including its CIFAR configuration, loss, LARS and recurrent warmup/cosine schedule. Source notices are preserved in adapted files and the pinned test fixtures. See also the [paper](https://arxiv.org/abs/2310.01012).

| Configuration | Loss arithmetic | Vector-gradient clipping | Role |
|---|---|---|---|
| `ssley-paper.json` | Ambient autocast, as released | None | Author-source alignment; numerically unstable in the observed run |
| `ssley-stable.json` | FP32 with autocast disabled | Joint norm 1 on one-dimensional parameters | Successful full-budget stability variant |
| `ssley-fp32-only.json` | FP32 with autocast disabled | None | Controlled clipping ablation |

The stable recipe clips only BN affine parameters and biases excluded from LARS adaptation, after AMP unscaling. Matrix gradients retain the reference LARS update. Invalid vector gradients remain untouched so GradScaler can skip the update. This deliberately changes optimization. All three configurations keep the architecture, augmentations and learning-rate schedule aligned with the pinned source.

| Setting | Value |
|---|---|
| Encoder | ResNet-18, CIFAR 3x3/stride-1 stem with padding 2, no initial max pool |
| Projector | 512 -> 2048 -> 2048 -> 2048; hidden BN/ReLU; Linear biases enabled |
| Training | 1000 epochs; batch 256; 195 framework steps per epoch; seed 7 |
| Optimizer | Author-fork LARS; LR 0.3 encoder/projector, 0.1 classifier; momentum 0 |
| LARS options | eta 0.02, clip_lr, exclude_bias_n_norm; weight decay 1e-4 except classifier 0 |
| Scheduler | 10-epoch warmup from 3e-5, recurrent step-wise cosine to zero |
| Precision | Network FP16 mixed; loss precision depends on selected configuration |
| Augmentation | Bicubic crop scale 0.2–1; jitter 0.4/0.4/0.2/0.1 with p=0.8; grayscale p=0.2; flip p=0.5; solarization p=0.1 on each view; no blur |
| Evaluation | Detached online linear classifier; top-1/top-5 on the test split |

The author fork adapts 2D zero-decay classifier weights, unlike the existing library LARS branch; `ssley_reference_lars.py` preserves that behavior locally. Training uses CIFAR standard deviations (0.2470, 0.2435, 0.2616), while evaluation follows the released rounded values (0.247, 0.243, 0.261). Seed 7 is our explicit choice, not a recovered paper seed.

## Environment and preparation

The recorded successful environment used Python 3.12.3, torch 2.8.0+cu126, torchvision 0.23.0+cu126 and Lightning 2.6.6 on one RTX A6000, with 8 CPUs and 32 GB host RAM allocated. Install a CUDA-compatible torch/torchvision build for your cluster, then from the repository root:

```bash
python -m pip install -e '.[dev]'
python -m pip freeze > server-environment.txt
```

Prepare the dataset once on a network-enabled host:

```bash
python benchmarks/cifar10/ssley-paper.py   --config benchmarks/cifar10/server/ssley-stable.json   --data-dir /absolute/data/cifar10 --output /absolute/runs/data-preparation   --download-only
```

`--download-only` exits before training. Compute jobs use `download=False`.

## Run and resume

In an allocated GPU session:

```bash
python benchmarks/cifar10/ssley-paper.py   --config benchmarks/cifar10/server/ssley-stable.json   --data-dir /absolute/data/cifar10 --output /absolute/runs/ssley-stable-seed7   --accelerator gpu --workers 4 --threads 4
```

For Slurm, adapt the partition/GPU request to your permitted allocation and set:

```bash
export SPT_REPO=/absolute/path/to/stable-pretraining
export SPT_PYTHON=/absolute/path/to/venv/bin/python
export SPT_DATA=/absolute/data/cifar10
export SPT_CONFIG=benchmarks/cifar10/server/ssley-stable.json
export SPT_OUTPUT=/absolute/runs/ssley-stable-seed7
sbatch benchmarks/cifar10/server/hydra-ssley.sbatch
```

The launcher finds the newest checkpoint under the output directory and stops before its 12-hour wall-time limit. The callback resets the session timer at each allocation while retaining the training schedule. No automatic resubmission is configured. Check that no existing job uses the output directory before resubmitting. `SPT_STOP_AFTER_EPOCHS=50` can bound an initial allocation without changing the 1000-epoch schedule; unset it to complete training.

For manual resume, add `--resume /absolute/path/to/last.ckpt` with the original configuration and output directory. Recipe mismatches are rejected. Prefer epoch-boundary resume; mid-epoch recovery can replay data and is not a bitwise replay. Runtime counters are per invocation, while history/numerical-event files append across resumptions.

The no-clipping comparison starts fresh with `ssley-fp32-only.json`, its own output directory and `--stop-after-epochs 50`. It does not resume from the stable run. Compare its trajectory with the stable run's first 50 epochs, not only the stable run's final 1000-epoch score.

An optional short real-data preflight is available as `hydra-smoke.sbatch`. The Python `--smoke` flag instead uses synthetic data only for wiring checks; neither is a representation-quality experiment. Neither flag was used for the final result.

## Outputs and observed result

- `resolved.json`: actual recipe, launch arguments and framework versions.
- `history.jsonl`: completed epoch metrics; `progress.json`: current step.
- `run-status.json`: running, paused, complete or failed. Only complete denotes the full budget.
- `numerics.jsonl`: AMP skips and invalid-loss events. The guard stops damaged model state, invalid validation metrics or 32 consecutive non-finite loss batches.
- `checkpoints/last.ckpt`: checkpoint on clean exit; periodic checkpoints are nested under `runs/.../checkpoints/`.
- `backbone.pt`: exported encoder on clean exit.

The stable run completed 1000 epochs at **89.46% online top-1 / 99.63% top-5**, with finite epoch metrics and verified final artifacts. The no-clipping run collapsed at epoch 28 and stopped during epoch 29 due to non-finite BatchNorm buffers. See the [final report](../results/ssley-cifar10.md) for numerical events, provenance, limitations and the distinction from strict paper-score reproduction.
