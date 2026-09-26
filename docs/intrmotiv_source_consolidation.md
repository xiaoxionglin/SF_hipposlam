# Runtime source integration

The canonical NEMO checkout is `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam`.
The GitHub default branch is `master`. New development starts from its latest
commit; running and queued jobs retain their qualified source paths. Do not
switch the NEMO canonical checkout while jobs are using it.

## September 26 integration

The integration combines the September 15 consolidation, CA3 state-goal and
z-relation follow-up (`2da7c742`), predictive active goals (`a28ffee4`),
task-general transfer (`b279cc94`), cued reward transfer and calibrated frozen
DG controls (`20fbcbe0`), and uncommitted corridor, landmark, and replay-retention
fixes. The canonical vault study package is synchronized at version 1.12.0.
Existing experimental modes remain opt-in and checkpoint/configuration defaults
are preserved. The evaluator now defaults to the active corridor workspace.

Overlapping changes are reconciled in the shared batched manager, rather than
restoring the obsolete scalar planner. Both manager paths use the same
cue-specific reward-goal selector. Terminal coverage advances without falsely
classifying an unavailable final pose as out of bounds.

NEMO source snapshots, content hashes, job dependencies, and unpublished commit
history are preserved under
`/work/classic/fr_xl1014-corridor-geometry/IntrMotiv/source_merge_20260926/`.
The final integration bundle preserves the local merge history. Published
merge parents preserve the existing GitHub branch histories; source-only work
is incorporated in the resulting tree. No training output is moved or deleted.

Validation covers pinned Black/isort/Flake8, runtime and workflow tests, world
model tests, and a mixed-cue regression through both batched planner paths.
The desktop suite passes 702 tests with two environment-dependent skips.

For the next integration, inventory commits and dirty files first, compare
against the prior inventory, and merge only new lineages in an isolated
checkout. Normalize formatting before three-way comparison. Run pinned hooks
with normal process access if Black stalls in the restricted process
environment. Verify the published tree SHA against the tested local tree.

## September 15 consolidation (historical)

## What was integrated

- The latest qualified controller lineage from
  `SF_hipposlam_controller_cpu_selected_20260914`, including stored replay,
  checkpoint retention, transport, controller diagnostics, and regression tests.
- The canonical vault study workflow (1.8.1), associated declarative studies,
  independent DDQN baseline/evaluation modules, and existing adapters.
- Current shared Git history through `00802a22`, preserving the newer device-side
  topological manager and PPO batch-validity fix.
- The depth sensor's opt-in `--depth_sensor_inverse=True` response. The default
  remains legacy behavior; inverse mode uses a fixed gain of 10.

Controller changes were merged against the organized parent `ce85f564` with
formatting normalized before three-way comparison. Python AST equality was
used to identify changes that were formatting-only. Historical prototype
variants were preserved rather than activated alongside the qualified runtime.
New baseline launcher copies now use the canonical checkout; archived launchers
retain their historical paths.

## Preservation and active jobs

The inventory records 37 folders: the canonical checkout and 36 alternatives.
All distinct nonignored source variants (373 content hashes, approximately 5 MB
before compression) were preserved before editing the canonical runtime.
Complete checkout backups, including ignored files, symlinks, and Git metadata,
are stored in the allocated workspace:

`/work/classic/fr_xl1014-train/IntrMotiv/source_retirement_20260915/`

- `inventory.json`: original per-folder Git state and per-file SHA-256 hashes.
- `source_variants.tar.gz`: content-addressed differing source files.
- `checkouts/`: complete backups verified with a checksum-based rsync dry run.
- `jobs.txt` and `script_dependencies.json`: job/source dependency evidence.
- `retirement.json`: final per-folder retirement decisions, written at retirement.

Three source folders are retained while jobs reference them:

| Folder suffix | Jobs at inventory |
| --- | ---: |
| `controller_cpu_selected_20260914` | 24 |
| `controller_rr1_20260913` | 12 |
| `controller_stored_production_release_20260912` | 8 |

The other 33 alternatives were retired after fresh job checks and an empty
checksum-based comparison against the full backups. Retained release source
hashes were rechecked after removal and were unchanged.
No training output or checkpoint directory is removed. Preserve Git worktree
administration used by retained copies; do not blindly prune registrations.

## Repeatable procedure

1. Inventory actual files as well as commits: many historical folders have
   uncommitted changes or no Git metadata.
2. Trace current Slurm working directories and scripts before editing sources.
3. Archive complete checkouts in the allocated workspace and checksum-verify.
4. Merge the qualified lineage into an isolated staging checkout, preserving
   newer shared commits and using the canonical study package.
5. Run the pinned pre-commit hooks and the IntrMotiv/workflow tests on desktop
   and NEMO with bounded CPU threads; keep NEMO temporary files in the workspace.
6. Push a new branch without rewriting shared history, update the canonical
   checkout, and verify matching commit IDs.
7. Retire only checksum-verified, inactive alternatives; retain an explicit
   record for the live releases and retire them after their jobs finish.

Avoid creating a checkout for each small fix. Use one development branch and
only freeze an additional release checkout when a live run needs its exact
source. Track its jobs and retirement condition at creation.

## Validation

Desktop validation of the integrated source passed all three pinned pre-commit
hooks and 583 runtime, workflow, and world-model tests. NEMO verification passed
573 tests with 10 CUDA tests skipped. The desktop editable-install check passed
all 17 depth tests after synchronization. The published source tree exactly
matched the tested tree before retirement.

Retirement completed for 33 inactive folders; three live release folders remain.
Their shared publication-worktree metadata is locked against pruning. Full
retirement results are recorded in `retirement.json` in the workspace archive.
Desktop and NEMO use `codex/nemo-consolidation-20260915`. Shell push credentials
were unavailable, so the connected GitHub app published the tested tree; the
local pre-publication commit is also preserved in the archive bundle.
