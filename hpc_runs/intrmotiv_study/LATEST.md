# Latest Standardized Workflow

- Implementation: `1.8.1` (local; NEMO2 remains `1.7.1`)
- Study schema: `intrmotiv/study/v1`
- Canonical package: `hpc_runs/intrmotiv_study/`
- NEMO2 runtime copy: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/hpc_runs/intrmotiv_study/`
- Canonical guide: `04_implementation/standardized_study_workflow.md`
- Reference study: `hpc_runs/studies/graph_stabilized_recruitment.study.json`

## 1.8.1 checkpoint target discovery

Checkpoint discovery now passes the StudySpec's telemetry and intervention
targets to the existing NEMO2 selector. The old selector defaulted to historical
5M/25M/50M/75M/100M targets, causing nonstandard preflights and the 150M/300M
production targets to fail `render-telemetry`. The selector retains its original
default for legacy callers. Synchronize both `intrmotiv_study/telemetry.py` and
`evaluation/build_place_field_sweep.py`; focused tests cover custom and late
horizon targets. Deployed to the isolated `SF_hipposlam_controller_compatibility_20260912`
checkout: 35 focused canonical/target/audit tests pass remotely. The original
shared checkout remains at 1.7.1. The real one-row 327,680-frame manifest rendered
successfully; ordinary Slurm evaluator job 8057320 completed its rollout and
produced validated DG, worker and pre-threshold maps across six episodes.

## Deployment status

Version 1.8.0 is staged locally: optional `analysis.loader_backend: "process"`
uses spawned workers for TensorBoard parsing; the default remains `"thread"`.
Both backends preserve row order and shared-window semantics, and the CLI
reports each completed run. All 43 focused tests pass locally, including
real-event process/thread equivalence and error propagation. Not yet deployed
or benchmarked on NEMO2; synchronize and rerun tests there before use.

Version 1.7.1 fixes exact run discovery for the standard nested launcher layout
`RUN_/00_RUN`: an empty outer container is excluded when its declared experiment
is nested inside. Ancestors with their own config, summary, or checkpoint payload
and distinct duplicate directories still fail as ambiguous. This is an
analysis/discovery-only fix; training code and study fingerprints are unchanged.

Synchronized and tested on NEMO2 on 2026-09-11: 41 tests passed locally and
remotely (canonical, common-window, repeat-8, and DG-capacity suites). Separately,
the DG-capacity study's analysis tag paths and grouping were corrected after
checking actual TensorBoard tags. Its revised fingerprint is recorded in the
interim report; submission audit confirms the original training commands.

Version 1.7.0 adds `collect-online --latest-common`: it loads each run once,
uses the latest step covered by every declared run and metric, and applies
`analysis.terminal_width` to that common endpoint. This mode disables scalar
reservoir sampling and fails on missing histories or empty/nonfinite window
means. Existing explicit-window and per-run-terminal modes remain available.
The fixed-reward repeat-8 study is the reference application; its StudySpec and
fingerprint are unchanged. See the canonical guide for the repeatable command.
Synchronized to NEMO2 on 2026-09-11; all 35 canonical, common-window, and
repeat-8 tests passed both locally and on NEMO2.


Version 1.6.0 accepts sorted, unique intervention checkpoint targets and
requires exactly one row per selected condition, seed, and target. This supports
the DG-capacity study's 75M and 300M intervention panels (54 rows). Single-target
studies retain their existing row contract. Local canonical/study suite: 30 tests
passed; NEMO2 synchronization verification is recorded in the DG-capacity launch
record.

Version 1.5.0 adds optional `telemetry.intervention.where` selection by validated
RunSpec context. Selected runs receive the intervention checkpoint in the
standard inventory even when their seeds are outside the ordinary field-map
subset. This supports all-seed goal probes without evaluating flat policies as
goal-conditioned policies. Local and synchronized NEMO2 focused suites:
27 tests passed. The updated IntrMotiv runtime passed 257 tests. Ten CA3-memory
training preflight jobs completed with exit 0 and passed the 2M-frame scientific
runtime audit. Exact deployment and production records are in
`06_experiments/ca3_memory_novelty_goal_implementation.md`.

The previous vault source and NEMO2 runtime copy were synchronized at `1.4.1`. Version
1.4 extends the compact online-spatial snapshot contract with cached place-field
and graph diagnostics while retaining compatibility with older 1.x studies and
v1 snapshots. The synchronized NEMO2 suites passed 26 workflow tests and 196
IntrMotiv tests. Ordinary Slurm preflight job `7983229` completed with exit
`0:0`; its workspace NPZ retained the configured 4,096-sample artifact window,
the scalar series used the latest 1,024 samples, both graph summaries were
present, and no image series was written.
For a later version, synchronize the complete package and rerun:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest hpc_runs.test_intrmotiv_study
python -m hpc_runs.intrmotiv_study validate \
  hpc_runs/studies/graph_stabilized_recruitment.study.json
```

The full-data collector benchmark was intentionally stopped once synchronization
and contract verification were complete. Runtime acceleration is secondary to
reducing repeated Codex discovery and code generation.

## Version policy

- Patch: compatible bug fix with no study-file changes.
- Minor: backward-compatible component or optional field.
- Major: incompatible schema or artifact contract. Introduce a new schema ID
  and migration notes; do not silently reinterpret existing studies.

Update this file, `version.py`, tests, and the guide together when the standard
changes, then synchronize and test the NEMO2 runtime copy.

## 1.4.1

- Increased the default milestone artifact window to 100,000 behavior samples
  while retaining a separately configurable latest-10,000 scalar window.
- Replaced repeated concatenate-and-truncate buffering with a fixed circular
  buffer and recorded the effective scalar window in new snapshots.
- Retained full backward compatibility with existing v1 snapshots and custom
  smaller preflight windows.

## 1.4.0

- Added the canonical 5M, 25M, 50M, 75M, and 100M online snapshot milestones.
- Added occupancy-normalized multilevel field components, mono-field and peak
  separation diagnostics, complete graph buffers, prospective edge outcomes,
  reliable global efficiency, grounded controllability, and cached detailed
  graph diagnostics to optional v1 snapshot fields.
- Added `collect-spatial --include-details` for per-unit, per-field, and
  directed graph-edge CSVs without another DMLab rollout.

## 1.3.0

- Added `intrmotiv/online-spatial/v1` validation and shared online/offline
  spatial calculations for 19×19 occupancy-corrected DG maps and segmented
  trajectories.
- Added `collect-spatial`, which discovers exact StudySpec run identities,
  writes per-snapshot, condition, and seed CSVs, preserves the study SHA-256,
  and renders figures only for explicitly selected runs and targets.
- Added readable selected-run DG contact sheets and occupancy/trajectory
  panels. Training never renders or uploads images.

## 1.2.0

- Added optional `target-control-intervention-v1` manifest generation. The
  intervention rows are selected from the standard checkpoint inventory and
  must contain exactly one declared checkpoint for every study run.
- Added the controllability study’s provenance-aware plan and runtime-gate
  auditors as thin study-specific adapters.

## 1.1.0

- Added bounded parallel TensorBoard loading, configurable with
  `analysis.max_workers` and defaulting to four workers.
- Added `audit-submission` for exact matrix, command, job-ID, and workspace-path
  validation against real Sample Factory `jobs.tsv` files.

## Full-system controller qualification

The active clean R5 qualification uses the isolated source
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam_controller_stable_head_20260912`
with workflow 1.8.1. Study: `hpc_runs/studies/full_system_controller_preflight_r5.study.json`;
SHA `b68edd0bcb84fd25d2779013e17511c2a97dfb0624741f0a56e48343a6d0d4d9`.
Its 379 runtime and 5 controller-audit tests pass remotely; canonical print-only
and submitted audits pass. Both PPO preflights completed and passed DG, frozen
trunk and spatial telemetry gates. R4 is debugging evidence only after optimizer
ownership drift was found. The optional collated-input performance candidate
was not deployed. All six exact GPU reload checks passed in job 8057366; the four DDQN runs
resumed as 8057362–8057365 and have advanced beyond their saved checkpoints.
Follow the current experiment record for the final 2M production gate; do not
reuse historical R4 launch helpers.

## Approved decoder-only waypoint revision

The user replaced goal-write waypoint F64 with goal-independent worker memory
and the existing target-ID FiLM decoder. See
`04_implementation/decoder_only_worker_goals_20260912.md`. This supersedes the
waypoint architecture in the earlier controller qualification section.
Three new 2M preflights use `full_system_controller_decoder_preflight.study.json`
(SHA `581955c17e808cffb2a038bbbdaa21da70eb13d6ea9aab6b264124fb5bca5159`),
jobs 8057437–8057439, isolated `SF_hipposlam_controller_decoder_only_20260912`
source. Fifty-one focused runtime/workflow tests and submission audits pass.
The direct R5 cells are unchanged and retained through
`full_system_controller_direct_qualification.study.json` (SHA
`fa9df8f17b3bcc1acad2f03458e20621e5ec3274f6afb9f9f32a7f9d26539b85`).
Require both three-cell qualifications before the revised 18-run production
matrix. Do not launch the superseded goal-write matrix.

Use the ordinary SF launcher `--submit` after print-only generation.
`resume_slurm_submission.py` operates on pending rows and skips rows still marked
`generated`; it is not the first-submission entry point for a print-only manifest.

Decoder-only checkpoint qualification: new waypoint PPO completed 2,048,000
frames with fixed trunk, learned DG/decoder and finite 1M/2M snapshots. Both
DDQN arms stopped at 327,680 with identical 1,023 main updates; HER added
13,077 positions. They resumed as 8057450/8057451. All three exact GPU reloads
passed in 8057452. See the experiment record for the remaining full 2M gates.
