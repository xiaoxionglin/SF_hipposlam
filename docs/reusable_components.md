# Shared components and project boundaries

SF_hipposlam owns the shared training implementation. Project repositories own
their questions, StudySpecs, evidence and reports. Reuse a small component from
here instead of copying an entire experiment directory into another project.

This integration was checked against SF_hipposlam `14ddead`, IntrMotiv
`5bc2f5b` and the standalone SimGaY source `bc2bcb2` on 4 November 2026.
It preserves existing defaults and makes new runtime behavior opt-in. Local
compatibility tests establish implementation parity, not scientific performance
or qualification for a cluster run.

## Where code belongs

| Location | Owner and purpose |
| --- | --- |
| [`sf_shared/`](../sf_shared/) | Small reusable components: DG projection, sampled depth, and offline checkpoint loading. Import a specific module; the package does not eagerly load training or vision code. |
| [`sample_factory/`](../sample_factory/) | Common training engine, configuration and optimizer lifecycle. |
| [`sf_working_directories/IntrMotiv/`](../sf_working_directories/IntrMotiv/) | IntrMotiv model, learner, evaluation and project-specific contracts. |
| [`sf_working_directories/default/`](../sf_working_directories/default/), [`jannek/`](../sf_working_directories/jannek/) | Existing model variants. Their decoder/core differences remain explicit. |
| [`sf_xxl/dmlab/`](../sf_xxl/dmlab/) | Existing HippoSLAM/world-model entry point. Its `train_hipposlam.py` currently imports the **default** model/parser; do not infer the selected implementation from the entry-point folder alone. |
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
| **GeoMem** | Its existing categorical DG implementation matches IntrMotiv. `CategoricalDGEncoder` can now import `sf_shared.dg` lazily; the supervised sensory runner exposes the same choice. The local backend remains available without installing SF. Its APPO adapter already uses IntrMotiv components; its restricted sequence core is not a substitute for the full research core. |
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
| Shared DG in GeoMem | Python: `projection_backend="sf_shared"`; supervised CLI: `--dg-projection-backend sf_shared` | `local` | Shared backend needs this SF checkout on Python's path. Local backend has no new dependency. Checkpoint metadata records the selection. |
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

## Verification

Focused tests cover exact shared/legacy outputs, gradients, running statistics,
state keys, old-checkpoint loading, explicit false parsing, optimizer restart,
partly/wholly skipped PPO updates, workflow provenance and scientific metric
invariants. SimGaY tests retain golden environment/event sequences, independent
actuator randomness, training isolation and manager checkpoint validation.
GeoMem compares prediction and one optimizer step across both backends.

The pre-existing StudySpecs keep their fingerprints and rendered rows. These
checks do not deploy the code to NEMO2, rerun scientific experiments, demonstrate
GPU parity, or qualify altered scientific conditions for production.

For future reuse, add the narrow component here, keep its consumer's old path
selectable when behavior changes, and test the saved configurations/checkpoints
that the affected projects actually use. Keep provenance beside the component
and update this guide when ownership changes.
