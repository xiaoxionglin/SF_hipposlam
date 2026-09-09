# IntrMotiv Study Workflow

This directory is the canonical, latest reusable implementation for defining
and evaluating IntrMotiv studies in this vault.

The synchronized NEMO2 runtime copy is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam/hpc_runs/intrmotiv_study/`.
Keep its version and file contents aligned with this canonical copy before
using new workflow features on the cluster.

Current implementation version: `1.4.1`; current schema:
`intrmotiv/study/v1`.

The version constants in `version.py` are authoritative. Do not copy this
package into a batch-specific directory. Extend it here, add focused tests, and
keep existing study files valid whenever possible.

A study records the workflow version it was authored against. Compatible
newer implementations can still load it; a study authored for a newer or
different-major implementation is rejected.

## Included components

| Module | Responsibility |
| --- | --- |
| `spec.py` | Validate a study, expand its Cartesian product, render names and arguments, and compute its SHA-256 fingerprint. |
| `study.schema.json` | Provide an editor- and tooling-friendly description of the v1 JSON structure. |
| `sample_factory.py` | Convert a complete study into Sample Factory `Experiment` and `RunDescription` objects. |
| `discovery.py` | Resolve every declared run directory exactly once in a single batch-tree scan. |
| `submission.py` | Audit a generated `jobs.tsv`, commands, job IDs, and workspace paths against the study. |
| `tensorboard.py` | Locate expected run directories and collect terminal or fixed-window metrics with bounded parallel loading. |
| `analysis.py` | Produce reusable mean/SD/count summaries and explicit within-seed linear contrasts. |
| `telemetry.py` | Use the authoritative checkpoint selector and generate full, trajectory, and optional intervention manifests. |
| `spatial_contract.py` | Define and validate compact online snapshots and share place-field/trajectory calculations with training. |
| `spatial.py` | Collect exact-study spatial snapshots and optionally render selected post-hoc figures. |
| `cli.py` | Expose validation, rendering, online collection/analysis, and telemetry-manifest generation. |

Run the CLI from the repository root:

```bash
PY=/home/fr/fr_xl1014/.conda/envs/SFgit/bin/python
"$PY" -m hpc_runs.intrmotiv_study validate hpc_runs/studies/STUDY.study.json
"$PY" -m hpc_runs.intrmotiv_study render-runs hpc_runs/studies/STUDY.study.json
"$PY" -m hpc_runs.intrmotiv_study collect-spatial \
  hpc_runs/studies/STUDY.study.json SNAPSHOT_ROOT OUTPUT_DIR --include-details
```

Use this SFgit interpreter by default on NEMO2 for tests, analysis, and figure
rendering; it contains the project scientific stack, including Matplotlib.

The complete convention and NEMO2 lifecycle are documented in
`04_implementation/standardized_study_workflow.md`.

## Extension rule

Add generally useful behavior to this package. Keep truly study-specific
logic in a thin adapter that consumes `StudySpec` or its generated artifacts.
Create a separate workflow only when the study cannot preserve the v1 run,
analysis, or telemetry contracts; document the incompatibility before doing
so.

When an established Python base factory cannot yet be expressed as a complete
argument bundle, pass an explicit `experiment_builder(run)` callback to
`build_run_description`. The callback may assemble the base command, but it
must preserve the study-generated run names and ordering.
