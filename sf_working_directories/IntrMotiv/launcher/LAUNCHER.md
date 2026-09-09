# IntrMotiv Slurm Launcher

IntrMotiv uses Sample Factory's existing `sample_factory.launcher.run` Slurm backend. The project wrapper only supplies the stable NEMO2/SFgit resource profile.

Dry-run a batch first:

```bash
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.hrl_intrinsic_arch_search \
  --print-only
```

Submit the same run description:

```bash
sf_working_directories/IntrMotiv/launcher/launch_nemo2.sh \
  sf_working_directories.IntrMotiv.dmlab.experiments.hrl_intrinsic_arch_search \
  --submit
```

Each invocation receives a timestamped directory under:

```text
train_dir/_slurm/<batch-name>/<submission-time>/
```

That directory contains generated sbatch scripts, `submission.json`, `jobs.tsv`, `scancel.sh`, and separate `logs/*.out` and `logs/*.err` files. Training artifacts retain Sample Factory's existing layout under `train_dir/<batch-name>/...`.

The template remains at `dmlab/experiments/nemo2_sfgit_intrmotiv.sh` so previous launch commands continue to work. New batches normally require only a new `RunDescription` module and a unique `BATCH_NAME`; the NEMO resource profile does not need to be copied or edited.

Resource defaults can be overridden with normal Sample Factory launcher arguments. `SLURM_WORKDIR` overrides the generated work directory and `SFGIT_PYTHON` overrides the launcher interpreter.

The template sends per-job temporary files, XDG/Torch and Matplotlib caches,
and W&B cache/staging to
`/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/runtime`. Override this
root with `INTRMOTIV_RUNTIME_ROOT` when necessary. DMLab does not derive its
level-cache path from these environment variables, so run descriptions that
enable the level cache must explicitly provide a workspace
`--dmlab_level_cache_path`.

## Source control on NEMO2

The authoritative model checkout is
`/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam`; the desktop code checkout is
legacy. Obsidian vault synchronization does not commit this model repository.
The September 9, 2026 baseline is on
`codex/intrmotiv-nemo2-baseline-20260909` in
`https://github.com/xiaoxionglin/SF_hipposlam`.
The reviewed integration branch intended to become the next `master` is
`codex/intrmotiv-integration-20260909`. It retains the complete IntrMotiv
runtime and canonical study workflow while excluding unrelated Jannek changes,
duplicate top-level IntrMotiv tests, and a hardcoded legacy Slurm template.
Shared Sample Factory extensions use generic opt-in contracts; IntrMotiv sets
those contracts in `maybe_overwrite_rnn_size`.

Before a new study or implementation change, inspect `git status --short --branch`
and `git log -5 --oneline` here. Commit each coherent, validated change with
explicit source paths and a descriptive message; push the active branch after
each completed task and before production submission. Review `git diff --cached
--stat` and `git diff --cached --check` before committing. Check that local HEAD
matches `git ls-remote origin refs/heads/$(git branch --show-current)` after
pushing. Do not assume vault sync or a clean tracked-file diff captures new files:
always inspect untracked paths too.

Before pushing Python changes, run `pre-commit run --all-files`. The GitHub
workflow deliberately checks the complete tree on every push, so running only
against staged files can miss an existing repository-wide failure. Black and
isort cover all Python source. Flake8 excludes the inherited
`sf_working_directories/default/`, `sf_working_directories/jannek/`, and
`sf_xxl/` trees; maintain new shared, `hpc_runs/`, and IntrMotiv code without
adding further broad exclusions.

GitHub's core test and coverage workflows mirror the NEMO2 runtime with Ubuntu
and Python 3.10. They install the base package plus ONNX support and run tests
that do not require licensed ROMs, proprietary simulators, GPUs, or a display.
Simulator integration remains covered by the NEMO2 suite below, where the
production environment and DeepMind Lab build are available. Keep dependency
installation separate from these simulator requirements so hosted package or
runner changes do not masquerade as model-code failures.

Keep model source, tests, declarative studies, evaluation scripts, and launcher
code in Git. Keep checkpoints, rollouts, caches, logs, and generated submission
folders in the allocated workspace. Loose source backups and conflict-recovery
copies remain on disk but are ignored. The small, documented archive under
`hpc_runs/source_snapshots/` is retained as historical study provenance; it is
not the current runtime.

The cleaned integration branch passed 333 IntrMotiv, workflow, and launcher
tests (23 deprecation warnings) with:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/fr/fr_xl1014/.conda/envs/SFgit/bin/python -m pytest -q \
  --import-mode=importlib -p no:cacheprovider \
  sf_working_directories/IntrMotiv/tests hpc_runs/test*.py \
  tests/test_launcher.py
```

Use the importlib mode because the test directories contain repeated module
names. These CPU tests do not establish training quality or replace a Slurm
preflight. For future tests, direct temporary outputs to the allocated workspace.

The baseline review used the existing tests, a syntax/credential-pattern scan,
and the documented source-archive SHA-256. Direct inspection on NEMO2 avoided
a broad export of untracked files; `rg` was unavailable there, so use Python or
`grep` for remote inspections. No runtime algorithm was changed while organizing
this baseline.

### Publishing when NEMO2 lacks GitHub credentials

The initial baseline push over NEMO2 HTTPS failed with `could not read Username`.
The desktop's existing GitHub SSH access worked. Keep NEMO2 authoritative: use
a temporary bare Git repository on the desktop to fetch the already reviewed
NEMO2 branch and push it with desktop SSH. This does not involve the legacy
desktop model working tree or transfer untracked NEMO2 files.

```bash
# Run on the desktop; choose a fresh temporary directory.
publish_dir=$(mktemp -d /tmp/intrmotiv-publish.XXXXXX)
git init --bare "$publish_dir"
git -C "$publish_dir" fetch git@github.com:xiaoxionglin/SF_hipposlam.git \
  refs/heads/master:refs/heads/master
branch=codex/intrmotiv-integration-20260909
git -C "$publish_dir" fetch \
  nemo2:/home/fr/fr_xl1014/SF_git_XXL/SF_hipposlam \
  "refs/heads/$branch:refs/heads/$branch"
git -C "$publish_dir" push git@github.com:xiaoxionglin/SF_hipposlam.git \
  "refs/heads/$branch:refs/heads/$branch"
```

For future branches, replace `branch` with the reviewed NEMO2 branch. After
publishing, fetch that branch from origin on NEMO2, set its upstream, and compare
HEAD to the remote branch SHA. Never copy tokens or private keys between hosts.
