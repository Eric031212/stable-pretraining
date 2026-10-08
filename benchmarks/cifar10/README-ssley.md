# SSL-EY on CIFAR-10

This contribution integrates SSL-EY from [Unconstrained Stochastic CCA: Unifying Multiview and Self-Supervised Learning (ICLR 2024)](https://arxiv.org/abs/2310.01012) into stable-pretraining with a loss, composable forward function and method class.

## Library interface

```python
import torch
import stable_pretraining as spt

model = spt.SSLEY(
    backbone=spt.backbone.from_torchvision("resnet18", low_resolution=True),
    projector=torch.nn.Sequential(
        torch.nn.Linear(512, 2048),
        torch.nn.BatchNorm1d(2048),
        torch.nn.ReLU(),
        torch.nn.Linear(2048, 2048),
    ),
    optim={"optimizer": {"type": "AdamW", "lr": 1e-3}},
)
```

The example above demonstrates the generic API, not the complete experiment recipe. For custom composition use `spt.Module(forward=spt.forward.ssley, ssley_loss=spt.losses.SSLEYLoss(), backbone=..., projector=..., optim=...)`.
Training accepts exactly two views as a list or named mapping, with `image` tensors in each view. Labels are optional and never enter the SSL objective. Evaluation returns backbone embeddings without running the projector. The formal recipe separately trains a detached online linear classifier.

For centered projections X and Y, the loss is `-2 trace(C) + trace(V @ V)`, where `C = 2 X.T @ Y / (N - 1)` and `V = (X.T @ X + Y.T @ Y) / (N - 1)`. Negative losses are expected. The generic loss promotes half/bfloat16 covariance arithmetic to FP32 by default and retains float64 inputs. Autograd-aware gathering requires equal local batch shapes; use `drop_last=True`. Multi-GPU runtime validation is not claimed.

## Reproducible experiment

See the [server recipe guide](server/README.md) for the single-GPU ResNet-18 experiment, pinned author provenance, configuration and resume commands. The successful run uses `server/ssley-stable.json`; the original-arithmetic recipe and no-clipping ablation are separately named configurations.

| Experiment | Completed epochs | Online top-1 | Online top-5 | Outcome |
|---|---:|---:|---:|---|
| FP32 loss + vector-gradient clipping | 1000 | 89.46% | 99.63% | Completed and final model verified |
| FP32 loss, no clipping | 28 of planned 50 | 10.62% | 56.74% | Numerical failure during epoch 29 |

These are final recorded epoch scores on the 10000-image CIFAR-10 test split, not best-epoch selections. The ablation failed and is not a matched-duration accuracy comparison against the 1000-epoch run. At epoch 28 the stable run scored 73.32%.

Read the [full experiment summary](results/ssley-cifar10.md), [machine-readable results](results/ssley-cifar10.json), and [training curves](results/ssley-cifar10.png). FP32 loss and vector-gradient clipping change the released arithmetic/optimization, so this is a **stability variant**, not strict reproduction of the paper's 91.43% result.

## Checks

```bash
python -m pytest stable_pretraining/tests/test_ssley.py stable_pretraining/tests/test_ssley_recipe.py stable_pretraining/tests/test_ssley_reference.py stable_pretraining/tests/test_ssley_runlog.py -q
```

Coverage includes mathematical values/gradients and invariances, invalid input handling, low precision, label-free training, forward layouts, Manager/checkpoint integration, recipe semantics, detached classification, vector clipping, AMP recovery and failure guards. Pinned fixtures additionally check author loss/gradient parity, initialization, two optimizer updates and the complete 195000-step learning-rate schedule. CUDA-specific cases skip without CUDA. Source parity does not establish identical historical training trajectories.
