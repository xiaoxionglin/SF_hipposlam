# NEMO source consolidation (2026-09-15)

The canonical NEMO checkout is `/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam`.
Use this checkout for new work. Existing running and queued jobs keep their
qualified source paths until they finish.

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

The other 33 alternatives are retirement candidates. Retirement must recheck
current jobs and verify the full archived contents immediately before removal.
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
hooks and 583 runtime, workflow, and world-model tests. NEMO verification and
retirement results are recorded in the workspace archive before completion.
