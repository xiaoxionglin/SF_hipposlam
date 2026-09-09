# Encourage DG-Regularizer Factorial

## Purpose

`intrmotiv_encourage_dg_regularizers_20260825` is the controlled successor to
the anti-collapse batch. It restores the previously healthy encoder setting:
`encoder_reward_method=encourage` and `encoder_batch_loss=True`. The previous
global-punishment and row-repulsion results were confounded by disabling that
batch usage signal and switching to `mean` or `punish` feedback.

## Fixed Configuration

- Fixed ImageNet-pretrained `layer2_resnet18`; trainable DG projection and
  BatchNorm state.
- `F=16`, `R=8`, `L=64`; fixed-length no-reward DMLab openfield.
- Simultaneous updates, one CPU policy per job, no PBT, 100M environment steps.
- HRL uses the policy-buffer graph, 10k global option-event half-life,
  novelty-only target selection, and `hit_distance` worker reward.
- Flat and HRL runs have identical encoder settings and seeds.

## Factorial

There are 60 runs: 2 architectures x 2 thresholds x 5 loss arms x 3 seeds.

| Factor | Values |
| --- | --- |
| Architecture | `flat`, fixed/global HRL |
| DG threshold | 2.43, 2.20 |
| Loss arm | control, global 0.01, global 0.03, row 1.0, global 0.01 + row 1.0 |
| Seed | 8, 99, 123 |

The row coefficient is increased from the previous 0.01 because that loss
fell to approximately `1e-7` late in training. Coefficient 1.0 keeps its
early magnitude near the existing batch-usage auxiliary rather than creating
an arbitrary broad sweep. The global 0.03 arm is retained as a controlled dose
test under the recovered encoder setting. No global 0.03 plus row 1.0 cell is
run because it is an unnecessarily aggressive interaction before either
individual mechanism has supporting evidence.

## Validation and Analysis

The preflight module contains four 1M-step jobs that cover both architectures,
both thresholds, global 0.03, row 1.0, and the combined arm. It must show
finite nonzero intended regularizer losses, active `batch_reward_loss`,
nonzero DG activity, and HRL option diagnostics before production submission.

The learner logs DG duty-cycle minimum/mean/maximum and normalized usage
entropy in addition to density, silent-unit fraction, multi-activation rate,
and pre-threshold statistics. Interpret a regularizer as useful only if it
improves distributed DG use and 10k rollout maps without degrading flat
coverage or HRL target-hit and option-success behavior.

After training, build a 140-task deterministic place-field manifest:

- all three seed finals for every architecture/threshold/loss condition;
- five checkpoints near 5M, 25M, 50M, 75M, and 100M for seed 99.

Each evaluation has 10,000 policy decisions. Artifacts include thresholded
DG-event maps and continuous pre-threshold logit maps. The latter expose a
spatial signal that remains below the event threshold.

## Commands

```bash
cd ~/SF_git_XXL/SF_hipposlam

sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.encourage_dg_regularizers_preflight \
  --print-only

sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.encourage_dg_regularizers \
  --print-only
```

Submit only after inspecting the generated manifests and successful preflight
metrics. All train directories, Slurm logs, checkpoints, W&B local files, and
evaluation outputs belong under `/work/classic/fr_xl1014-train`.
