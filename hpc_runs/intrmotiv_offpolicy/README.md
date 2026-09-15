# IntrMotiv recurrent DDQN/HER diagnostic

This package is a **frozen-reference local-control diagnostic**, not a completed
adaptive representation/manager architecture. The primary runtime supports the
selected F16 `dg_goal_input=none` parent. Write-conditioned memory and separate
target reconstruction have deterministic tests; write-conditioned collection
is explicitly rejected until actor-state rebuilding is integrated and tested.

The two arms share source decoder/FiLM weights, fresh Q initialization, permitted
observations, epsilon schedule, first-arrival reward, and nominal replay
intensity. They differ only in requested HER fraction (0 or 0.8). There is no
matched PPO learner in this package: parent zero-shot evaluation is a separate
reference, not a controlled PPO-versus-DDQN algorithm comparison.

- `checkpoint.py`: parent action/layout validation and conversion inventory.
- `features.py`: exact source normalization, frozen canonical DG, depth and map
  bypass; first-batch source-head parity check; no pose feature route.
- `contracts.py`: recognition, shift/injection, finite-budget reward, Double DQN.
- `replay.py`: bounded stream/episode replay with 71-step prefix validity,
  delayed future-HER eligibility, original-goal fallback and replay RNG state.
- `worker.py`, `batch.py`: current-parameter memory rebuild for both networks,
  masked Huber targets, hard copies including write modulation.
- `terminal.py`: explicit final-observation validity and deep-copy before reset.
- `train.py`: source environment, parallel collection, child counters,
  checkpoints and standard scalar layout. The physical episode decision clock
  is normalized by 1,800 without clipping; option budget by 64. These are input
  scales, not assertions about an engine's exact final decision number.
- `evaluate.py`: readout callback into the established exact-start matched
  evaluator; evaluation patch is supplied separately with regression tests.
  Grounded spatial destination qualification remains required.

Replay currently stores exact frozen preactivations and bypass outputs, not
frozen trunk features. Therefore it deliberately cannot support DG adaptation.
The working/reference encoder is entirely frozen in this stage. An adaptive
stage must change this feature contract, retain fixed reference labels, and
rebuild working memories from trunk features. Reusing the cached preactivations
would silently make the representation optimizer ineffective.

Invalid terminal observations are stored as `None`, excluded from TD/HER, and
counted. The same-step collector still receives the real reset observation for
the next physical episode. No previous image is substituted as a successor.
The supplied DMLab patch explicitly marks its cached terminal image invalid.

Checkpoint files omit replay content and are labeled
`warm_restart_requires_refill`; they are not exact-resume artifacts. The current
CLI refuses reusing an existing run namespace and does not implement restart.
Replay's own state-dict contract preserves storage and sampling RNG for a future
checksummed sidecar integration.

Run the independent tests with the project interpreter:

```bash
/home/xiaoxiong/miniforge3/envs/SF_git/bin/python -m unittest hpc_runs.test_intrmotiv_offpolicy
```

The runtime must be isolated from existing jobs. Use the reviewed terminal
contract patch and planner optimization in the selected new source checkout.
The deployment and submitted preflights are recorded in
`06_experiments/intrmotiv_ddqn_her_implementation_20260911.md`.

## v2 finite-budget repair

The current child model uses `intrmotiv/ddqn-worker/v2` and a shared nonlinear
clock-aware readout. V1 children require the preserved v1 evaluator; there is no
implicit migration. HER attempts retain the first-hit suffix through at most 64
steps; `PositionBatcher` carries splits with their original deadline and delivers
exactly 256 valid TD positions/update. Target-copy cadence and accepted-decision
update cadence are explicit CLI settings. Unknown environment remaining time is
`None`; certified final observations are read from outer vector autoreset fields.
See `06_experiments/intrmotiv_ddqn_her_v2_repair_20260911.md` for the boundary
bootstrap table, numerical qualification, v1 evidence and staged v2 studies.

For future repairs, run archived audit assertions only against archived source.
Compare actual runtime hashes, test the vector-info selector as well as reset,
and verify nonlinear expressivity on both local and cluster Python builds.
Preserve failed fitting and short-budget cadence measurements. Do not promote
from TD loss alone; require matched-command evaluation and per-goal coverage.

## Optional batched execution

`--learner-execution=batched` batches frozen physical-prefix reconstruction and
time×batch decoder evaluation. Default remains `reference`. It is qualified for
the inspected row-independent TargetFiLMDecoder and goal-independent writes;
requalify after decoder changes. The objective, TD budget and v2 state dict stay
unchanged. See `06_experiments/intrmotiv_ddqn_throughput_20260911.md` for paired
runtime evidence, limits and SF integration choices.

For repeated throughput work, use `profile_runtime` with the canonical StudySpec
and existing scalar counters before adding instrumentation. Use `benchmark`
with the actual runtime decoder/checkpoint, then an isolated paired end-to-end
preflight and `audit_runtime`. Learner-only speedup does not establish overall
FPS or scientific improvement. Preserve exact accepted-decision/update budgets
when adding async collection; off-policy DDQN needs no PPO policy-lag filter.

## Native Sample Factory execution

Use the existing `train` entry point with `--execution-backend=sample_factory`.
The qualified CPU topology is `--num-envs=32 --sf-workers=16 --sf-splits=2
--sf-rollout=32 --sf-batch-size=1024 --sf-async=true --torch-threads=8
--sf-learner-threads=1 --learner-execution=batched`. All parent, replay and
learning-budget arguments remain the same. `standalone` remains the default.

This reuses SF's ParallelRunner, RolloutWorker, InferenceWorker, Batcher,
BufferMgr, ParameterServer/Client, double buffering, backpressure, summaries,
W&B integration and Slurm launcher. The upstream DQN PR is not required: the
installed fork already exposes a custom learner factory. DDQN loss, recurrent
replay, HER, PositionBatcher and checkpoint/evaluation schemas are reused here.

`sf_buffers` adds one opt-in policy-output shape. Existing SF copying code carries
that feature/command packet; no SF source files are modified. `sf_transport`
restores per-stream physical order because SF merges trajectory slices by buffer
index. A nonterminal rollout tail waits for the next actor feature packet. It
never uses a reset image or re-encodes a duplicate successor. Transport tails at
the requested training endpoint are counted separately from consumed frames.

The adapter supports the existing frozen, goal-independent worker and fixed
frames per decision. Certified terminal images, multi-policy/PBT and exact
resume remain unsupported and fail closed where encountered. Checkpoints retain
the v2 worker contract and use the existing manifest-driven evaluator. Native
SF improves execution; it does not implement adaptive DG or a new manager.

Reuse `audit_runtime` for final TD/update/transport accounting and SF's built-in
worker profiles for bottlenecks. Test serial and actual multiprocessing with
`hpc_runs.test_intrmotiv_sf_native --smoke-output /tmp/UNIQUE_NAME [--serial]`;
the local test also needs the installed SF source on PYTHONPATH. The desktop
sandbox may block `torch_shm_manager`; use the ordinary permitted runtime rather
than replacing SF shared-memory transport. Seed the model factory explicitly:
this fork's inference-process initialization does not seed Torch itself.

W&B qualification (2026-09-12): fresh child configs now clear the parent's
`wandb_unique_id`, group and tags before SF initializes logging. SF then creates
an independent run ID. Keep the regression test when changing parent-config
copying. The actual compute-node smoke run completed and its `ddqn/*` metrics
and `train/env_steps` were verified through the W&B API; installed SDK support
alone is not evidence of a successful upload. The production record is
`06_experiments/intrmotiv_ddqn_sf_production_20260912.md`.

## Dashboard parity gate (2026-09-12)

Native SF execution must also connect application telemetry: register IntrMotiv's
summary router and call its existing `TrainingSpatialTelemetry` coordinator.
`sf_telemetry.py` reconstructs thresholded activity from frozen preactivations,
uses the existing bounded spatial ring/NPZ contract, and forwards the environment's
otherwise discarded batched episode statistics. It must not use exclusive
recognition masks as DG activity or expose inherited parent graph state as a
learned child graph. Missing graph diagnostics are unavailable, not zero.

Qualify dashboard changes with `intrmotiv_ddqn_telemetry_preflight.study.json`:
check actual W&B tags for DG, spatial/trajectory and episode coverage, a valid
100k-sample NPZ, and unchanged runtime budget/frozen-reference gates. The original
native production batch omitted these spatial artifacts; completed training
history cannot be backfilled from checkpoints. Old DG-capacity dashboards remain
in their original W&B project, while DDQN runs are in `IntrMotiv`. See
`06_experiments/intrmotiv_ddqn_metric_consistency_20260912.md` for the comparison
and exact artifact locations. Prefer config/cloud metadata over rescanning all
historical TensorBoard events simply to locate dashboards.
