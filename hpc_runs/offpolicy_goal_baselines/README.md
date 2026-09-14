# Off-policy visual goal baselines

This package implements observation-matched CRL+ and L3P+ baselines for the
IntrMotiv DMLab environment. It is standalone because the source algorithms are
off-policy replay methods, whereas Sample Factory's IntrMotiv learner is
on-policy APPO. The established Sample Factory Slurm launcher is still reused
for run expansion, workspace safety, independent jobs, and submission audit.

## Method provenance

- CRL source: Eysenbach et al., *Contrastive Learning as Goal-Conditioned
  Reinforcement Learning*, and the official
  `google-research/google-research/contrastive_rl` implementation.
- Stable CRL choices: Bortkiewicz et al., *Accelerating Goal-Conditioned RL
  Algorithms and Research* (JaxGCRL): L2 energy, symmetric InfoNCE, and
  log-sum-exp regularization.
- L3P source: Zhang, Yang, and Stadie, *World Model as a Graph: Learning Latent
  Landmarks for Planning*, and the official
  `LunjunZhang/world-model-as-a-graph` implementation.

The implementations are clean adaptations rather than vendored source. The
official trainers assume continuous actions, coordinate goals, HER, and
SAC/DDPG-style replay. IntrMotiv instead has eight discrete navigation actions
and egocentric RGBD observations.

## Matched contract

Both cells use:

- the authoritative ImageNet-pretrained `ResNet18Layer2` trunk, frozen in eval
  mode with all parameters having `requires_grad=False`;
- the same map, eight-action interface, action repeat, replay, future-goal
  relabeling, low-level categorical controller, and trainable post-trunk heads;
- no ground-truth pose as model or planner input;
- pose only for coverage and achieved-goal evaluation.

CRL+ learns the action-conditioned energy

$$
f(s,a,g)=-\lVert \phi(s,a)-\psi(g)\rVert_2^2/\tau
$$

with symmetric InfoNCE. L3P+ adds a directed temporal-distance head, a separate
symmetric temporal landmark embedding, farthest-point landmark medoids drawn
from achieved observations, a reachability-filtered sparse directed graph, and
Floyd-Warshall planning. Medoids replace the original decoded coordinate
centroids so every landmark remains a realizable frozen-visual goal. The
controller commits to a selected landmark for its predicted travel time and
excludes the immediately failed landmark on replanning, following the source
method's anti-sticking mechanism.

Long DMLab episodes are streamed into replay as ordered 512-decision segments.
This preserves valid future-goal pairs while allowing learning to begin before
the environment's 120-second timeout. The L3P-only landmark objective is not
evaluated or optimized in the CRL cell.

## Validation and launch

```bash
python -m unittest hpc_runs.test_offpolicy_goal_baselines
python -m hpc_runs.intrmotiv_study validate \
  hpc_runs/studies/offpolicy_goal_baselines_parallel_strong_preflight.study.json

hpc_runs/offpolicy_goal_baselines/launch_nemo2.sh \
  hpc_runs.offpolicy_goal_baselines_parallel_strong_preflight --print-only
```

Always run the canonical `audit-submission` command on the generated `jobs.tsv`
before submission. Training outputs, TensorBoard events, metrics, checkpoints,
Slurm logs, and runtime caches stay under `/work/classic/fr_xl1014-train`.

## Evaluation

The short preflight verifies runtime correctness only. A performance claim
requires matched seeds and reports at least coverage AUC, unique coverage,
goal-hit rate measured from telemetry, contrastive retrieval accuracy, policy
entropy, and wall-clock/sample throughput. L3P+ must additionally show that a
landmark graph was rebuilt and used without nonfinite path costs.
