# IntrMotiv Evaluation

This package is the reusable, TensorBoard-first evaluation layer for IntrMotiv.
It is separate from `analysis/`, whose scripts and reports describe individual
historical batches.

## Current Core

The initial core reads existing TensorBoard event files and writes:

- `per_run_terminal.csv`: one terminal-window row per policy run;
- `family_terminal_summary.csv`: family means, standard deviations, and counts;
- `condition_terminal_summary.csv`: schedule, half-life, and reward-method cells;
- `diagnostic_report.md`: conclusions supported by existing scalars and the
  causal probes that are not observable retrospectively;
- `manifest.json`: schema version, source batches, and observed metrics.

The terminal window is the final `min(10M frames, 20% of observed frames)`,
with a 1M-frame minimum. TensorBoard is authoritative; W&B upload state is
irrelevant to the analysis.

Run it from the Sample Factory repository with the `SFgit` interpreter:

```bash
PYTHONPATH=. /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m \
  sf_working_directories.IntrMotiv.evaluation \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_flat_iterative_baseline_nopbt_20260820 \
  /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/intrmotiv_hrl_persistence_comparison_20260821 \
  --output /work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/evaluation_recent_batches_20260824 \
  --workers 4
```

All analysis outputs belong in the allocated workspace, never the source
checkout or NEMO home directory.

TensorBoard event parsing is CPU-bound. `--workers 4` is a reasonable offline
default for a completed batch; use `--workers 1` on a constrained login node.

## DG Place Fields

`place_fields.py` is a checkpoint rollout evaluator for current DG activity.
It records DMLab position telemetry, hooks the shift-register core, and uses
the first register slot of every feature as the current DG activation. It
writes occupancy, occupancy-corrected rate maps, per-DG spatial information,
and active fraction. For example:

```bash
PYTHONPATH=. /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  sf_working_directories/IntrMotiv/evaluation/place_fields.py \
  --run-dir /work/.../run_dir \
  --out-dir /work/.../analysis/place_fields_example \
  --max-num-frames 10000 --checkpoint-rank 0
```

It is intentionally a real DMLab rollout, so run it through Slurm rather than
on the login node. Use `--save-raw-activations` only when downstream analyses
need the rollout matrix; it is substantially larger than the normal map
artifact.

On NEMO2, submit manifest rows as ordinary independent Slurm jobs with
`submit_place_field_sweep.py`. The command is print-only unless `--submit` is
present, and it writes a job-ID manifest when it submits. Do not use a Slurm
array for new telemetry sweeps: ordinary jobs are evaluated more reliably by
the current NEMO2 scheduler. For example:

```bash
PYTHONPATH=. /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest /work/.../analysis/place_fields_example/analysis_manifest.tsv \
  --output-dir /work/.../analysis/place_fields_example \
  --row 0-6

# After reviewing the printed ordinary sbatch commands:
PYTHONPATH=. /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  sf_working_directories/IntrMotiv/evaluation/submit_place_field_sweep.py \
  --manifest /work/.../analysis/place_fields_example/analysis_manifest.tsv \
  --output-dir /work/.../analysis/place_fields_example \
  --row 0-6 --submit
```

`run_place_field_sweep_single.sh` is the worker invoked by each job. The older
`run_place_field_sweep_array.sh` remains only as a compatibility wrapper for
already submitted and historical arrays.

This path is distinct from the legacy
`dmlab/experiments/run_generate_telemetry.py`. That file is still an unchanged
copy of Jannek's run description with `fr_js1764` paths and the older telemetry
entry point. The manifest evaluator keeps explicit checkpoint selection and
the current NPZ/pre-threshold-map contract, while adopting the same important
scheduling property: one ordinary Slurm job per evaluation.

`summarize_place_fields.py` reads existing `place_fields.npz` artifacts, with
no environment rollout. It writes `place_field_summary.csv` and a concise
Markdown report containing active fraction, spatial information, map
redundancy, and the number of distinct peak cells. By default it also writes
one clearly labelled `place_fields_<run>.png` grid per run plus a
`place_field_comparison.png` overview. The figures use the `Pillow` package
already present in `SFgit`; they do not alter the training environment:

```bash
PYTHONPATH=. /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  sf_working_directories/IntrMotiv/evaluation/summarize_place_fields.py \
  --input-dir /work/.../analysis/place_fields_example \
  --out-dir /work/.../analysis/place_fields_example
```

`analyze_place_field_manifest.py` is the reusable second-stage analyzer for a
replicated checkpoint manifest. It excludes silent units from thresholded-map
cosine and peak metrics, adds normalized peak-bin entropy and pairwise peak
distance, computes the same diagnostics for pre-threshold logit maps, and
writes terminal mean/standard-deviation and numerically ordered trajectory
tables:

```bash
PYTHONPATH=. /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python \
  sf_working_directories/IntrMotiv/evaluation/analyze_place_field_manifest.py \
  --input-dir /work/.../analysis/place_fields_example \
  --manifest /work/.../analysis/place_fields_example/analysis_manifest.tsv \
  --out-dir /work/.../analysis/place_fields_example/summary
```

The canonical end-to-end protocol, manifest/NPZ contracts, Slurm preflight,
interpretation rules, and extension points are documented in the IntrMotiv
Obsidian vault at `04_implementation/reusable_place_field_telemetry.md`.
Future analyses should extend these contracts compatibly instead of creating a
batch-specific rollout path.

Spatial information and map cosine are trajectory-conditioned descriptive
statistics. To make a stability claim, evaluate several checkpoints on the
same scripted coverage trajectory and add split-half and cross-checkpoint map
correlations.

### Architecture Trajectories

`build_place_field_sweep.py` writes the explicit architecture x checkpoint
manifest for the completed IntrMotiv batches. It selects the nearest retained
milestone to 5M, 25M, 50M, 75M, and 100M frames. The accompanying
`submit_place_field_sweep.py` creates one ordinary Slurm job per manifest row,
and `run_place_field_sweep_single.sh` evaluates that row.
`plot_place_field_trajectories.py` then creates a five-checkpoint DG-map contact
sheet per condition and family-level spatial-information/map-cosine
trajectories. All tools require an explicit workspace output directory.

## Current Limits

Existing online scalars cannot prove that a target match was intentional, that
DG units have stable place fields, or that graph deadlines are calibrated.
Those claims require the future checkpoint evaluator and bounded trajectory
artifacts described in `05_plans/reusable_evaluation_and_diagnostics_plan.md`.

Do not compare coverage measurements across `physical_episode` and
`telemetry_window` scope. The report records the source tag and scope for each
run so that this incompatibility remains visible.
