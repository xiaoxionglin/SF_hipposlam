# DG Anti-Collapse Batch

This batch tests mechanisms intended to reduce DG collapse and promote
landmarks outside heavily revisited observations. It does not add batch-wise
population, collision, density, usage, or multi-activation losses.

## Fixed Architecture

- ImageNet-pretrained, fixed `layer2_resnet18` trunk.
- Trainable DG projection and DG BatchNorm running statistics.
- `F=16`, `R=8`, `L=64`, fixed-length no-reward openfield.
- One CPU policy per Slurm job, no PBT, 32 workers x 2 environments.
- `encourage` is deliberately not included: this batch compares `mean` and
  `punish`, which were previously less useful at the high DG threshold.
- Worker reward is `hit_distance` for HRL.

## Main Factorial Sweep

The 108 main jobs are the Cartesian product of:

| Factor | Values |
| --- | --- |
| Architecture | flat, fixed/global HRL `policy_buffer` with 5k half-life |
| Encoder feedback | `mean`, `punish` |
| DG threshold | `1.8`, `2.0`, `2.2` |
| Global pre-threshold coefficient | `0`, `0.01`, `0.03` |
| Seed | `8`, `99`, `123` |

HRL is the previously strongest fixed/global candidate: no learned manager,
novelty-only target ranking, `hit_distance`, and a policy-buffer graph with a
5k global option-event half-life. It is not combined with a long-episode graph
or an iterative schedule in this focused experiment.

Each job runs for 80M environment frames. W&B project:
`SF_IntrMotiv_DGAntiCollapse`; groups separate flat and global-HRL conditions.

## Separate Weight-Only Arm

Twelve additional jobs hold the threshold at `2.43`, disable global
pre-threshold punishment, and set `dg_row_repulsion_coeff=0.01`. They span
flat/global-HRL, `mean`/`punish`, and the same three seeds. This arm is kept
separate because it tests an angular prior on DG rows rather than
observation-conditioned suppression.

## Hypotheses and Decision Criteria

Lowering the threshold should make `mean` and `punish` informative by
increasing DG events. The global pre-threshold penalty should reduce repeated
activations and rotate normalized DG rows away from represented locations.
Row repulsion should prevent redundant DG directions without using the current
batch. A useful condition must improve coverage or coverage AUC while retaining
noncollapsed DG activity; lower activity alone is not success.

Compare `intrmotiv/dg/density`, `silent_unit_fraction`, and
`pre_threshold_above_fraction` against coverage, occupancy entropy, target hit
rate, and intrinsic reward. Check checkpoint place fields before treating an
improved scalar coverage curve as a representation improvement.
