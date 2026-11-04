# Shared code: what changed and how to use it

Use this file as the practical guide to the 4 November 2026 organization and
reuse work. Start with [what changed](#what-changed), follow the
[usage examples](#usage-examples), and consult [enable and roll back](#enable-and-roll-back)
before changing an existing experiment. [Development checks](#development-checks-and-github-warnings)
explain how to avoid repeated formatting failures.

SF_hipposlam owns the shared training implementation. Project repositories own
their questions, StudySpecs, evidence and reports. Reuse a small component from
here instead of copying an entire experiment directory into another project.

This integration was checked against SF_hipposlam `14ddead`, IntrMotiv
`5bc2f5b` and the standalone SimGaY source `bc2bcb2` on 4 November 2026.
It preserves existing defaults and makes new runtime behavior opt-in. Local
compatibility tests establish implementation parity, not scientific performance
or qualification for a cluster run.

## What changed

| Change | Why it helps | Delivery |
| --- | --- | --- |
| Shared DG projection, sampled-depth adapters and CPU checkpoint loader in `sf_shared` | Projects can reuse the implementation without copying a whole encoder or importing the environment just to inspect a checkpoint. | [SF PR 3](https://github.com/xiaoxionglin/SF_hipposlam/pull/3), merge `667b887a` |
| Explicit behavior switches, real boolean parsing and separate SimGaY manager entry point | New options are selectable; old model/checkpoint paths remain usable. | SF PR 3 |
| Study workflow discovery and saved-map analysis in one package | Shared infrastructure stays with SF while study definitions and scientific evidence stay with their projects. | SF PR 3, workflow version `1.15.0` |
| Required workflow modules, schema and geometry maps included in wheels | A normal installation now contains dependencies previously visible only in a source checkout. | [SF PR 4](https://github.com/xiaoxionglin/SF_hipposlam/pull/4), merge `55f6f306` |
| Frozen starter files restored and protected by hash/inventory tests | Formatting cannot silently change an already distributed version. | SF PR 4; original ZIP and manifest retained |
| GeoMem backend choice connected to its built-in APPO representation | `--shared_dg_projection=True` now selects the shared implementation instead of being ignored. | [GeoMem commit `bb736ab`](https://github.com/xiaoxionglin/GeoMem/commit/bb736abb6302d5109b41d8d0b7726d649b9c3729) |

The workspace organization pass also clarified project entry points, moved
compseq's six authored scripts into `authoring/`, retained the two Office
workbooks in `local-copies/`, and removed duplicated GeoMem thesis prose while
preserving paper-specific priorities. On the original host, the complete
file-move and ownership record is
`~/Desktop/Projects/docs/audits/2026-11-04-organization.md`; the workspace entry
point is `~/Desktop/Projects/README.md`. These local files are outside this Git
repository. SAB and historical scientific results were excluded from changes;
the paper-controls checkout retained its scientific code and results.

SF's default branch is **`master`**. GeoMem's is **`main`**. A check passing on
one branch says nothing about a different branch's head commit.

## Where code belongs

| Location | Owner and purpose |
| --- | --- |
| [`sf_shared/`](../sf_shared/) | Small reusable components: DG projection, sampled depth, and offline checkpoint loading. Import a specific module; the package does not eagerly load training or vision code. |
| [`sample_factory/`](../sample_factory/) | Common training engine, configuration and optimizer lifecycle. |
| [`sf_working_directories/IntrMotiv/`](../sf_working_directories/IntrMotiv/) | IntrMotiv model, learner, evaluation and project-specific contracts. |
| [`sf_working_directories/default/`](../sf_working_directories/default/), [`jannek/`](../sf_working_directories/jannek/) | Existing model variants. Their decoder/core differences remain explicit. |
| [`sf_xxl/dmlab/`](../sf_xxl/dmlab/) | Existing HippoSLAM/world-model entry points. `train_hipposlam.py` imports the **default** model/parser; `train_dmlab.py` imports **jannek**. Check the executable's imports when choosing a model. |
| [`sf_examples/simgay/`](../sf_examples/simgay/README.md) | Optional completed-trial manager benchmark, with its own CLI and checkpoint contract. |
| [`hpc_runs/intrmotiv_study/`](../hpc_runs/intrmotiv_study/README.md) | Study validation, run discovery, collection and saved-output analysis, reusable by compatible projects. |
| `hpc_runs/studies/`, project repositories | Study-specific conditions, provenance, decisions and evidence. Frozen study inputs and exports remain pinned. |

The retained legacy DG class is an intentional fallback, tested against the
shared class. The four public depth classes are thin adapters so old imports
and state keys continue to work. Whole encoders and cores are not interchangeable.

## Project reuse review

| Project | Reusable material and integration decision |
| --- | --- |
| **IntrMotiv** | Main source of shared depth/DG/checkpoint components and workflow utilities. Runtime improvements already upstream include replay, checkpoint retention/milestones, geometry telemetry and graph caching; they are not recopied. Two previously separate recovery patches now have explicit opt-in flags. |
| **SimGaY** | The standalone high-level manager, trial history, losses, actuator environment and checkpoint protocol now live under `sf_examples/simgay`. Task metrics remain separate from model inputs. Full navigation, oracle transfer and ongoing project variants are not silently merged. |
| **GeoMem** | Its existing categorical DG implementation matches IntrMotiv. `CategoricalDGEncoder` can now import `sf_shared.dg` lazily; the supervised sensory runner and built-in APPO representation expose the same choice. The local backend remains available without installing SF. Its APPO adapter already uses IntrMotiv components; its restricted sequence core is not a substitute for the full research core. |
| **MorphSCA** | Component alignment and stability analysis are useful reference methods. The current small-factor matching assumptions (up to eight components) do not fit the DG/RL use case; no runtime dependency is introduced without a matching evaluation contract. |
| **SF_hipposlam_paper_controls** | Frozen paper-specific checkout and evidence owner. Shared ideas may be evaluated in development, but this integration does not rewrite its code, results or provenance. |
| **compseq / nextGen_Call** | Presentation authoring and submission assets retain their own owners. They do not supply an appropriate training dependency. |
| **Website_local / Office** | Website and administrative workflows remain separate; no training integration is justified. |
| **Mail bridge / bin / writings** | Local/private tools and reference aliases stay outside the scientific runtime. |
| **SAB** | Excluded from inspection and changes at the user's request. |

The IntrMotiv source was pinned before concurrent cue-transfer edits. A target
control patch was also left out: it changes the evaluation population, horizon
and censoring relative to newer focus-source/focus-target and physical-arrival
logic. That requires a versioned scientific protocol, not a structural merge.

## Enable and roll back

Append runtime flags to the same training command and retain its original
model, environment and scientific configuration. Enable one change at a time.

| Capability | Enable | Default / rollback | Scope and compatibility |
| --- | --- | --- | --- |
| Shared DG implementation | IntrMotiv: `--shared_dg_projection=True` | `False` selects retained legacy class | Same constructor, parameter/buffer names and strict state-dict loading; BatchNorm semantics remain whatever the saved config selects. |
| Shared DG in GeoMem | Python: `projection_backend="sf_shared"`; supervised CLI: `--dg-projection-backend sf_shared`; built-in APPO representation: `--shared_dg_projection=True` | `local`, or APPO `False` | Shared backend needs this SF package installed. Local backend has no new dependency. Custom APPO representation factories require the flag to remain False. Encoder exports and full actor state dictionaries load strictly in both directions. |
| Capped inverse depth | `--depth_sensor_inverse=True` | `False` restores pass-through sampling | Default/default-entry-point, jannek and sf_xxl parsers stay False; IntrMotiv's existing fresh-run default stays True. Old saved configurations retain their prior response. Requires fixed preprocessing, not running normalization of depth. |
| Preserve loaded optimizer rates | `--preserve_loaded_optimizer_lr=True` | `False` | During initialization only, keep the loaded optimizer's actual group rates. Fresh initialization, later scheduler updates and explicit config/PBT changes keep existing behavior. This does not promise an identical future training trajectory. |
| Recover unsupported PPO minibatches | IntrMotiv: `--recover_empty_ppo_minibatches=True` | `False` retains strict failure | Requires PPO with `distance_learning=True`, `double_value=False`. Skip minibatches with fewer than two valid decisions before minibatch forward/SGD. Empty epochs do not advance the scheduler; a wholly skipped update does not apply recruitment. Skips are logged and change the effective training sample population. |
| Exclude generated analysis folders from run discovery | `collect-online` / `render-telemetry`: `--run-discovery-mode training` | `legacy` | Avoid false duplicate matches beneath `analysis/`; genuine duplicate training directories still fail. Choice is recorded in analysis provenance, without changing study hashes. |
| Saved-map concentration analysis | Import `hpc_runs.intrmotiv_study.field_concentration` | Do not call it | NumPy-only analysis; existing collectors and NPZ schemas are unchanged. |
| Optional graph labels/export formats | `render-graph` CLI, or optional `render_graph_outcomes` arguments | Existing PNG/PDF renderer defaults | Explicit matrix keys and evidence labels; no new success-rate interpretation is imposed on an existing outcome matrix. See the package guide. |
| Standalone manager benchmark | `python -m sf_examples.simgay.train --help` | Continue using the existing navigation entry point | Separate executable; no global training switch. Compatible manager weights can be exchanged under the documented protocol; full navigation transfer remains unverified. |
| Existing DG world model | `--dg_world_model=True` on supported entry point | `False` | Already integrated before this pass. See [the world-model guide](../sf_xxl/dmlab/WORLD_MODEL.md). |

Boolean options in the four DMLab parameter modules now use the real boolean
parser: `--flag=False` disables the flag instead of interpreting the nonempty
string as True. Existing defaults and saved effective booleans are unchanged.
Old command lines that accidentally depended on the parsing bug must specify
the intended value explicitly.

When resuming, pass the rollback value explicitly: the saved config may contain
the enabled value. Keep the source checkpoint and use a separate output/run
directory for the comparison. Switching a behavior flag does not undo updates
already made to model weights, optimizer state or running statistics.

Minibatch recovery protects the SGD loop, not the entire incoming-batch path:
observation/return normalization, bootstrap preparation and environment-step
accounting still occur before or around it. If the selected summary minibatch
is skipped, that update may have no training summary; the skip warning remains.

The checkpoint-loader extraction and depth adapters are source-organization
changes with unchanged legacy behavior; they do not have redundant runtime
implementations. For a structural regression, revert the corresponding Git
commit or use the pre-integration revision. Do not overwrite an experiment's
saved provenance to make it look as if it ran with another implementation.

## Usage examples

### Install and choose an interpreter

Use the environment set up by the [installation guide](../README.md), then run
this from the SF_hipposlam repository root:

```bash
python -m pip install -e '.[analysis]'
python -m sf_examples.simgay.train --help
python -m hpc_runs.intrmotiv_study --help
```

The editable installation follows this checkout's code. Omit `-e` for a normal
installation. `analysis` adds Matplotlib for figure rendering; it does not
install DeepMind Lab or deploy anything to a cluster. Use the same interpreter
for installation, tests and execution.

### Select the shared DG implementation

For an existing IntrMotiv training command, add `--shared_dg_projection=True`.
Keep its other experiment arguments and use a new output directory. To compare
with the retained implementation, repeat from the same source checkpoint with
`--shared_dg_projection=False`. This changes code ownership, not the intended
DG computation; it does not automatically select a different learning rule.

GeoMem exposes the same choice in two workflows. From GeoMem's
`geomem-experiments/` directory, after following its installation instructions:

```bash
# Full APPO path; starts training when invoked.
python -m experiments.geomem.train --experiment=shared_dg_trial \
  --shared_dg_projection=True
```

Use `False` to restore the local APPO projection. For the independent supervised
sensory-prediction runner, this small example also starts training:

```bash
python experiments/minigrid_sensory_prediction.py \
  --dg-projection-backend sf_shared \
  --train-episodes 4 --test-episodes 2 --max-steps 24 --epochs 2 \
  --output-dir figures/shared_dg_smoke
```

Choose a new output directory and use `--dg-projection-backend local` to roll
back. See
[GeoMem's APPO guide](https://github.com/xiaoxionglin/GeoMem/blob/main/geomem-experiments/experiments/geomem/README.md)
and [sensory-prediction guide](https://github.com/xiaoxionglin/GeoMem/blob/main/geomem-experiments/experiments/minigrid_sensory_prediction.md)
for their different objectives and dependencies. The shared backend requires
this SF fork to be importable; custom APPO representation factories must keep
`--shared_dg_projection=False`.

### Inspect an existing checkpoint on CPU

Replace the path below with an actual IntrMotiv checkpoint:

```python
from pathlib import Path
import torch
from sf_shared.checkpoints import load_checkpoint_dict

checkpoint = load_checkpoint_dict(Path("path/to/checkpoint.pth"), torch.device("cpu"))
print(checkpoint.keys())
```

This reads the checkpoint using CPU memory mapping without constructing an
environment or starting training. It returns the saved dictionary; it does not
instantiate the model or change its parameters. See the dependency notes below
for the retained loader semantics.

### Try the standalone SimGaY manager

This short wiring check starts training and writes a new result directory:

```bash
python -m sf_examples.simgay.train --output train_dir/simgay_smoke \
  --session-batches 1 --num-envs 2 --eval-every 1 --eval-sessions 2
```

Choose a fresh `--output` path on each invocation. The short run is not a
learning-performance evaluation. Add `--actuator-success-rate 0.8` for the noisy
actuator condition and retain it when evaluating that condition. See the
[manager guide](../sf_examples/simgay/README.md) for checkpoint initialization,
frozen evaluation and the task/model defaults.

### Analyze saved study outputs

In these commands, replace `STUDY`, `BATCH_ROOT` and `NEW_OUTPUT_DIR` with your
study JSON, training-batch directory and a fresh analysis directory:

```bash
python -m hpc_runs.intrmotiv_study validate STUDY
python -m hpc_runs.intrmotiv_study collect-online STUDY BATCH_ROOT NEW_OUTPUT_DIR \
  --run-discovery-mode training
```

`training` avoids mistaking generated analysis directories for training runs.
Use `--run-discovery-mode legacy` for the original discovery behavior. Neither
command submits jobs or modifies the study definition. Graph rendering and
NumPy-only concentration analysis are described in the
[workflow guide](../hpc_runs/intrmotiv_study/README.md).

## Checkpoints and dependency boundaries

`sf_shared.checkpoints.load_checkpoint_dict` is the existing CPU/mmap offline
loader. Its historical import from IntrMotiv `evaluation.place_fields` still
works. NumPy allowlisting remains context-local and aliases/private copy-on-write
storage are preserved. General training checkpoint loading is unchanged.

`sf_shared.dg` needs Torch; `sf_examples.simgay` needs Torch and NumPy.
Importing these modules does not load DMLab, TorchVision or a training entry
point. Depth additionally uses the existing Sample Factory encoder base.
Consumers should import what they use rather than create another package-wide
registry or copy the shared source.

Install from this repository with `python -m pip install .`, or use
`python -m pip install '.[analysis]'` for Matplotlib-based saved-output rendering.
The wheel includes the reusable study workflow, runtime transport dependencies,
study schema and the two default geometry maps. Study-specific launch scripts,
source snapshots and result bundles remain in the source checkout. An installed
wheel is not a copy of the whole research workspace.

The versioned `exports/intrmotiv_prediction_starter/` directory is a frozen
distribution. Its contents must match its original manifest and ZIP byte for
byte; formatters exclude it. Make future changes as a new versioned export
instead of rewriting the published payload or its recorded hashes.

## Verification

Focused tests cover exact shared/legacy outputs, gradients, running statistics,
state keys, old-checkpoint loading, explicit false parsing, optimizer restart,
partly/wholly skipped PPO updates, workflow provenance and scientific metric
invariants. SimGaY tests retain golden environment/event sequences, independent
actuator randomness, training isolation and manager checkpoint validation.
GeoMem compares prediction and one optimizer step across both backends, checks
the APPO parser's actual selection, and loads encoder exports and full actor
checkpoints across both implementations.

CI builds a wheel and checks it from a temporary directory with source checkout
and editable-install paths removed. The probe verifies runtime imports, CLI help,
the packaged schema and both default geometry loaders. Frozen-export tests also
check file hashes, ZIP integrity and its complete member inventory.

The pre-existing StudySpecs keep their fingerprints and rendered rows. These
checks do not deploy the code to NEMO2, rerun scientific experiments, demonstrate
GPU parity, or qualify altered scientific conditions for production.

For future reuse, add the narrow component here, keep its consumer's old path
selectable when behavior changes, and test the saved configurations/checkpoints
that the affected projects actually use. Keep provenance beside the component
and update this guide when ownership changes.

## Development checks and GitHub warnings

Run these from the SF repository root before pushing Python changes:

```bash
python -m pip install pre-commit
python -m pre_commit install
python -m pre_commit run --all-files
git diff --check
```

The hooks use the versions and exclusions in `.pre-commit-config.yaml`.
Black/isort can modify files on their first run, which is reported as failure.
Review `git diff`, stage only the intended files, rerun the hooks, and commit
the resulting formatting before pushing. Running a formatter only in CI cannot
save those changes back to the branch. Use the configured hooks so the frozen
starter export stays excluded.

For changes to these reusable components, the focused source-checkout checks are:

```bash
python -m pytest -q tests/test_shared_dg.py tests/test_frozen_exports.py tests/test_simgay_*.py
python -m pytest -q hpc_runs/test_*.py
```

Use the project's test environment with pytest installed. Wheel packaging changes
also need the installed-wheel check in `.github/workflows/test-ci.yml`; passing
tests in an editable checkout alone cannot show that the wheel contains every
required module or data file.

When a GitHub failure email arrives, open its run and check the **branch and
commit** before comparing it with a successful check. On 4 November, run
[`41760343291`](https://github.com/xiaoxionglin/SF_hipposlam/actions/runs/41760343291)
failed Black/isort on the separate `codex/cue-transfer-20261104` branch. The
merged `master` revision `55f6f306` passed all four post-merge checks. A formatting
patch against cue-transfer `1cf64679` was prepared, passed the hooks plus 292
tests and 11 subtests, and was handed to its owning chat. That handoff is not a
claim that the branch repair has been pushed; check the branch's latest run.

The follow-up integration validation itself passed 380 SF/workflow/export tests,
10 GeoMem tests, installed-wheel checks and the PR checks. None of these counts
establishes scientific performance, GPU parity or deployment of the changes to
NEMO2.
