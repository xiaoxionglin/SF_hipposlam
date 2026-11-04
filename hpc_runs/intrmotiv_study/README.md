# IntrMotiv Study Workflow

This is the shared SF_hipposlam implementation for defining and evaluating
IntrMotiv studies. Version 1.15.0 integrates selected reusable changes from
IntrMotiv commit `5bc2f5b` (vault workflow 1.14.1), with legacy discovery kept
as the default. These are distinct implementation versions; this source update
does not establish deployment in any NEMO2 checkout or running job.

Current shared implementation version: `1.15.0`; current schema:
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
| `field_concentration.py` | Opt-in NumPy measurements of saved-map concentration; no default collector metrics or runtime changes. |
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

## Latest common training window

Use `collect-online STUDY BATCH_ROOT OUTPUT_DIR --latest-common` to align all
runs to their latest shared step and metric coverage in one read per run.
The study's `analysis.terminal_width` defines the window width. This is mutually
exclusive with `--window-low/--window-high`; default collection is unchanged.
See `04_implementation/standardized_study_workflow.md` for the repeat-8 example.

## Optional discovery and analysis controls

`collect-online` and `render-telemetry` accept
`--run-discovery-mode=legacy|training`. The default, `legacy`, preserves the
previous exact-name discovery and its ambiguity errors. Select `training` when
an `analysis/online_spatial/.../RUN` artifact directory shadows the real training
directory. It skips directories under `analysis/` within the supplied batch
root, while still rejecting multiple actual run directories elsewhere. Use
`legacy` for a layout that intentionally stores real runs below `analysis/`.
The mode is execution metadata, not a StudySpec field; existing study files,
training arguments, hashes, and checkpoint formats are unchanged. An explicit
training-mode choice is recorded in the generated analysis/study manifest.
`collect-spatial` reads snapshot files directly and does not use run discovery.

```bash
python -m hpc_runs.intrmotiv_study collect-online STUDY BATCH_ROOT OUTPUT_DIR \
  --run-discovery-mode=training
python -m hpc_runs.intrmotiv_study render-telemetry STUDY BATCH_ROOT OUTPUT_DIR \
  --run-discovery-mode=training
```

For an explicitly selected pair of saved count matrices, `render-graph` exposes
the graph renderer's optional label, canvas scale, and export formats:

```bash
python -m hpc_runs.intrmotiv_study render-graph counts.npz figures/graph \
  --attempts-key=attempts --successes-key=successes --title="Stored graph" \
  --ratio-label="Stored hits / attempts" --figure-scale=0.8 --format=svg
```

Choose matrix keys and a label matching the evidence. Counts must be finite,
nonnegative, aligned square matrices, with successes no greater than attempts.
Without export overrides, the Python renderer and CLI retain the original
10-inch canvas, prospective-hit label, and PNG/PDF save settings. Repeat
`--format` to request multiple formats. Existing spatial collection does not
generate additional graph figures automatically.

`field_concentration.calculate_field_concentration(rate_maps, occupancy)` is
an opt-in Python API for occupancy-corrected maps shaped `(rows, columns, units)`.
Its default support threshold is one observation per bin. Unvisited bins are
unknown, silent units have undefined shape, and concentration does not measure
geometric fragmentation. It adds no fields to existing NPZs or collector tables.
