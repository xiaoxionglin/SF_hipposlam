# Latest Standardized Workflow

- Implementation: `1.7.1`
- Study schema: `intrmotiv/study/v1`
- Canonical package: `hpc_runs/intrmotiv_study/`
- NEMO2 runtime copy: `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/hpc_runs/intrmotiv_study/`
- Canonical guide: `04_implementation/standardized_study_workflow.md`
- Reference study: `hpc_runs/studies/graph_stabilized_recruitment.study.json`

## Deployment status

Version 1.7.1 fixes exact run discovery for the standard nested launcher layout
`RUN_/00_RUN`: an empty outer container is excluded when its declared experiment
is nested inside. Ancestors with their own config, summary, or checkpoint payload
and distinct duplicate directories still fail as ambiguous. This is an
analysis/discovery-only fix; training code and study fingerprints are unchanged.

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
