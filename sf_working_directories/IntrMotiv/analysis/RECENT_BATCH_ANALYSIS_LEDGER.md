# Recent Batch Analysis Ledger

Last updated: 2026-08-21

This ledger is the source of record for completed analyses, provisional
findings, and work still required. TensorBoard event files, rather than W&B,
are the primary evidence source.

## Batches in scope

| Batch | Conditions | Analysis status | Frame coverage at this snapshot |
|---|---:|---|---|
| `intrmotiv_flat_iterative_baseline_nopbt_20260820` | 18 fixed-episode flat runs: feedback `{encourage, mean, punish}` x update schedule `{simultaneous, iterative}` x seed `{8,99,123}` | Analyzed | all 18 at 100.01-100.04M frames |
| `intrmotiv_hrl_persistence_comparison_20260821` | 18 fixed/global HRL, 18 long/per-stream HRL, 6 long flat controls | Provisional analysis; 8 complete and 34 still running | global HRL: 74.6-100.0M, stream HRL: 73.1-99.9M, long flat: 79.2-92.3M |

Preflights are implementation checks only. They are not included in outcome
comparisons.

## Completed Analysis

### Fixed-episode flat baseline

All 18 runs reached the requested 100M-frame target. Terminal-window means
use the final min(10M, 20% of observed frames) of each run.

| Feedback | DG density | Silent DG fraction | Coverage AUC | Unique cells | Interpretation |
|---|---:|---:|---:|---:|---|
| `encourage` | 0.0233 | 0.000 | 47.3 +/- 32.4 | 74.7 +/- 54.2 | Only HRL-compatible feedback: the DG population remains active, but coverage is variable. |
| `mean` | 0.00065 | 0.799 | 88.5 +/- 5.9 | 151.9 +/- 9.9 | High flat exploration coverage despite severe DG collapse. Invalid for DG-subgoal HRL. |
| `punish` | 0.00048 | 0.804 | 86.8 +/- 4.1 | 149.2 +/- 7.3 | Same conclusion as `mean`. Invalid for DG-subgoal HRL. |

Conclusion: DG activity is not itself an exploration objective. In this flat
control, `mean` and `punish` explore more broadly while their DG population is
mostly silent. `encourage` remains the correct fixed architecture choice for
HRL because it supplies landmarks, but it should not be described as the best
flat exploration rule. Iterative versus simultaneous updates show no
consistent coverage advantage across the three feedback rules.

### Persistence comparison: technical health

The late-training snapshots show that both HRL implementations run and form
nonempty graph statistics.

| Family | DG density | Silent DG fraction | Target hit rate | `T_ctrl` update rate | Known edge fraction | Coverage AUC |
|---|---:|---:|---:|---:|---:|---:|
| Fixed/global HRL | 0.0405 +/- 0.0049 | 0.0069 | 0.00249 | 0.00249 | 0.152 | 58.9 +/- 4.6 |
| Long/per-stream HRL | 0.0474 +/- 0.0131 | 0.0104 | 0.00243 | 0.000224 | 0.186 | 48.3 +/- 16.7 |
| Long flat control | 0.0356 +/- 0.0150 | 0.0000 | n/a | n/a | n/a | 34.0 +/- 28.9 |

Interpretation:

- DG activity no longer collapses under the fixed `encourage` configuration.
- Target hits remain sparse, about one per 400 policy transitions. This is too
  sparse to establish learned subgoal navigation from a terminal snapshot.
- The global graph is being updated as expected: its sampled target-hit and
  update rates agree, and its edge confidence is nonzero.
- The long-stream result is not yet evidence that long episodes improve
  exploration. Its periodic 900-step coverage telemetry is only comparable to
  the long flat controls, not to fixed-episode coverage.
- Global fixed HRL has higher and much less variable coverage than the matched
  fixed `encourage` flat runs in this snapshot, but the HRL jobs have not all
  reached 100M. Treat this as a hypothesis to test with matched seed/schedule
  windows, not a result.

### Half-life sensitivity status

The fixed/global half-life sweep is operating, but has no resolved winner:

| Global half-life | Coverage AUC | Hit rate | Known edge fraction |
|---|---:|---:|---:|
| 5k global option events | 58.1 +/- 4.6 | 0.00215 | 0.151 |
| 10k global option events | 59.7 +/- 4.0 | 0.00277 | 0.148 |
| 20k global option events | 58.8 +/- 5.8 | 0.00255 | 0.158 |

The three values overlap substantially. Do not select a half-life before all
runs finish and the seed-paired comparison is calculated.

The long/per-stream half-life sweep is **invalid** in this batch. Its commands
set `--hrl_persistent_fast_weights=False`. In the legacy per-stream update
path, that flag disables option-event decay, confidence-threshold gating, and
the running-average fast-weight update. Therefore its 5k, 10k, and 20k
arguments do not implement the intended per-stream half-life experiment.

This also explains the rate mismatch: the long branch updates `T_ctrl` only
when an arrival improves the stored time, whereas the global learner buffer
updates its running estimate on every intended success. The two graph modes
are consequently not yet semantically matched.

### Valid comparisons and update schedule

Valid now:

- All 18 fixed flat runs are valid for the feedback-rule and
  iterative-versus-simultaneous comparison.
- Fixed/global HRL is valid for the policy-buffer architecture and global
  half-life factor. At this update, 8/18 global runs are complete; its outcome
  comparison remains provisional until all complete.
- Long/per-stream HRL is valid only as a no-decay, per-stream long-episode
  graph condition. It can later be compared with long flat controls, but not
  used to estimate a fast-weight half-life response.

Invalid comparisons:

- Fixed-episode coverage cannot be compared directly to the long-run 900-step
  telemetry window. Compare global HRL only with fixed `encourage` flat, and
  stream-long HRL only with flat-long.
- Do not compare long 5k, 10k, and 20k as a half-life sweep.

There is no consistent iterative-update advantage. Late-window coverage AUC:

| Condition | Simultaneous | Iterative | Reading |
|---|---:|---:|---|
| Fixed flat `encourage` | 41.1 | 53.5 | Iterative is higher, but variance is high. |
| Fixed flat `mean` | 92.9 | 84.1 | Simultaneous is higher. |
| Fixed flat `punish` | 87.3 | 86.3 | Essentially tied. |
| Fixed/global HRL, averaged over half-lives | 59.6 | 58.1 | Simultaneous is slightly higher. |
| Long/per-stream HRL, averaged over nominal half-lives | 57.4 | 39.1 | Not interpretable as a half-life result; batch is incomplete. |
| Long flat | 24.2 | 43.9 | Iterative is higher, with only three seeds per cell. |

Use simultaneous updates as the default for the next HRL correction: it is
simpler and there is no replicated evidence that iterative updates improve
HRL. The provisional HRL candidate is fixed episodes plus a learner-owned
global graph, with 10k global option-event half-life and simultaneous updates;
it has the highest mean global coverage so far (61.5 over three seeds), but
must finish before it is selected.

## Remaining Analysis

1. Wait for all 42 persistence jobs to finish at 100M frames, then regenerate
   terminal summaries and record Slurm outcomes.
2. Make the primary comparison in common late windows, paired by seed and
   update schedule:
   - fixed/global HRL versus fixed `encourage` flat;
   - long/per-stream HRL versus long flat.
3. Plot coverage trajectories, target-hit trajectories, intrinsic-reward
   sparsity, graph confidence, and learned-deadline use. A terminal average
   cannot show whether worker navigation improved over training.
4. Repair the long-mode fast-weight semantics before treating any long
   half-life result as scientific evidence. Decay and running-average updates
   must be enabled independently of the RNN persistent-suffix mechanism.
5. Rerun the long half-life sensitivity study after that repair. The current
   18 long HRL runs remain useful for the long-episode/per-stream graph versus
   flat-long comparison, but not for the half-life factor.
6. Decide whether fixed/global HRL's apparent coverage advantage remains when
   compared at identical frame windows and matched seeds. Only then assess
   whether a learned manager or planning extension is warranted.

## Reproducibility

- Analyzer: `analysis/analyze_recent_batches.py`
- Raw terminal table: `/work/classic/fr_xl1014-train/IntrMotiv/SF_hipposlam/train_dir/analysis/recent_two_batches_20260821/per_run_terminal.csv`
- All values above are TensorBoard scalar means over each run's late terminal
  window. They are not W&B dashboard aggregates.
