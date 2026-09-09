# Recent Batch Statistics Report

Date: 2026-08-21

This report explains the statistics used in the fixed flat baseline and the
HRL persistence comparison. The numerical results are a late-training
snapshot: all flat runs are complete; the persistence comparison still has
running jobs. Use `RECENT_BATCH_ANALYSIS_LEDGER.md` for the current validity
status and remaining work.

## Data and Aggregation

The source data are TensorBoard event files. This avoids dependence on W&B
upload state.

For each run, `analyze_recent_batches.py` averages each logged scalar over its
late terminal window:

```text
terminal window = last min(10M frames, 20% of observed frames)
```

The tables then average those per-run terminal means. A `+/-` value is the
standard deviation across run conditions in that row, not a confidence
interval. In particular, an HRL family row contains repeated seeds across
half-lives, so it must not be read as 18 independent seed replicates.

All runs use `env_frameskip=8`: one policy decision normally advances eight
DMLab engine frames. Thus a 900-decision telemetry window is normally 7,200
engine frames.

## Spatial Coverage

### Cell definition

At policy decision `t`, the wrapper receives the first two coordinates of the
DMLab position telemetry and assigns the agent to one square cell:

```text
cell_t = floor(position_t[:2] / 100)
```

The grid width is 100 DMLab position units. Let `V_t` be the set of cells seen
from the beginning of the current episode or telemetry window through `t`, and
let `C_t = |V_t|`.

### Unique cells

```text
coverage_unique_cells = C_T
```

This is the number of distinct discretized cells visited in the measurement
interval. It ignores how often each cell is revisited.

### Coverage AUC

The logged AUC is not a geometric area. It is the time-average cumulative
number of cells discovered:

```text
coverage_auc = (1 / T) * sum_{t=1..T} C_t
```

The implementation accumulates `C_t` after each step and divides by the
number of steps. It rewards discovering cells early. For example, an agent
that reaches 20 distinct cells in the first 100 steps and then remains there
gets a much larger AUC than one that first reaches the same 20 cells at the
end, although both have identical final unique-cell counts.

For a fixed `T`, the AUC is bounded above by `T` only when every step visits a
new cell. It is not normalized to `[0, 1]` and should only be compared at the
same grid size, environment, and interval length.

### Occupancy entropy

If `n_c` is the number of steps in cell `c`, with
`p_c = n_c / sum_c n_c`, then:

```text
coverage_entropy = -sum_c p_c log(p_c)
```

Entropy measures how evenly time is spread across the cells already visited.
It is complementary to unique-cell count: an agent can have many cells with
low entropy if it spends nearly all its time in one of them.

### Fixed episodes versus long windows

The statistic has two different scopes in the current experiments.

| Run family | Measurement interval | Reset behavior | TensorBoard namespace |
|---|---|---|---|
| Fixed flat and fixed/global HRL | One physical fixed-length DMLab episode | Environment and coverage counters reset at terminal | `policy_stats/avg_z_..._coverage_*` |
| Long/per-stream HRL and long flat | 900 policy decisions | Only window counters reset; environment and RNN state do not | `intrmotiv/exploration/window/coverage_*` |

The long-window metric measures local exploration over each 900-decision
period. It does **not** report cumulative map coverage from the beginning of
the long episode, because its visited-cell set is cleared after every emitted
window. The fixed metric is conditioned on physical respawn; the long metric
is not. Therefore, compare fixed/global HRL only with fixed `encourage` flat,
and long/per-stream HRL only with long flat controls.

### Sample Factory smoothing

Each completed fixed episode or emitted long window is forwarded through the
standard Sample Factory episodic-statistics path. Sample Factory keeps up to
`stats_avg=100` recent reports per policy and logs their mean. The values in
TensorBoard are therefore already trailing averages. The terminal analysis
adds a second average over late logged snapshots. They are stable outcome
summaries, not raw single-episode measurements.

## DG Activity Diagnostics

All DG statistics are learner-minibatch diagnostics, computed from the DG
head output `a` after thresholding (`a > 0`). They are not direct map-place
field measurements.

| Metric | Definition | Interpretation |
|---|---|---|
| `dg_density` | Mean of `1[a > 0]` over sequence positions and DG units | Fraction of active DG units per sampled transition. |
| `dg_multi_activation_rate` | Fraction of sampled transitions with more than one active DG unit | Measures non-one-hot events; not necessarily an error. |
| `dg_silent_unit_fraction` | Fraction of DG units never active anywhere in the sampled minibatch | Detects within-minibatch collapse. It is not a lifetime silent-unit rate. |

The fixed flat result illustrates why these must be read together with
exploration: `mean` and `punish` have high coverage but roughly 80% silent DG
units. They may make reasonable flat policies but cannot provide usable DG
subgoals.

## Worker Reward and HRL Events

### Flat decoder reward

For non-HRL runs, the decoder reward is derived from the temporal internal
distance `d_t`:

```text
r_flat,t = reward_scale * (L + R - 1 - d_t)
```

Here `L=64`, `R=8`, and `reward_scale=0.1`, so the baseline is 71. This is a
dense temporal-distance signal and is not an external exploration score.

### HRL `hit_distance` worker reward

For HRL runs, the worker reward is target-gated. With target-hit indicator
`h_t`, target reward 1.0, distance-bonus coefficient 0.1, and the same
baseline:

```text
r_HRL,t = h_t * [1.0 + 0.1 * max(0, 0.1 * (71 - d_t))]
```

Thus a non-hit has exactly zero worker reward. `intrinsic_nonzero_fraction`
is consequently expected to be close to `target_hit_rate`; both quantify the
sparsity of worker supervision rather than map coverage.

| HRL metric | Exact sampled quantity |
|---|---|
| `active_target_fraction` | Fraction of valid replay transitions whose stored target ID is nonempty. |
| `target_hit_rate` | Mean stored `target_hit` flag, so hits per sampled valid policy transition. |
| `option_timeout_rate` | Mean stored option-expired flag, so timeouts per sampled valid policy transition. |
| `option_success_fraction` | `hits / (hits + timeouts)` for the learner minibatch. This is per completed option, unlike hit rate. |
| `tctrl_update_rate` | Mean stored `tctrl_updated` flag. It measures graph updates per sampled transition, not edge quality. |

At the snapshot, target hits are about `0.0024-0.0025`, or one hit per roughly
400 policy transitions. This is sufficient to populate graph entries but not
yet evidence that the worker has learned reliable subgoal navigation.

## Controllability Graph Statistics

The graph uses `F=16` DG landmarks. Diagonal edges are excluded from graph
fractions.

| Metric | Definition | Caution |
|---|---|---|
| `node_visit_weight_mean` | Mean DG visit weight | Its scale depends on graph scope, event count, and decay; do not compare it as an outcome score. |
| `known_edge_fraction` | Fraction of off-diagonal pairs with `T_ctrl > 0` and confidence above threshold | A nonzero edge says an arrival was recorded, not that it is useful or causal. |
| `forgotten_edge_fraction` | Seen edges below confidence threshold | Meaningful only when confidence decay is enabled. |
| `known_controllability_time_mean` | Mean `T_ctrl` over known edges | Travel-time estimate in policy decisions. Compare only within matched graph semantics. |
| `edge_confidence_mean` | Mean off-diagonal confidence | Fast-weight support, not probability of reachability. |

For the learner-owned global graph, one accepted rollout updates model buffers
outside autograd. On each global option completion/timeout, strengths decay by
`gamma = 0.5^(1 / half_life_options)`. On success from `i` to `j` in `tau`
steps, confidence and arrival time follow the weighted running update:

```text
C_ij <- gamma * C_ij + 1
T_ij <- (gamma * C_old_ij * T_old_ij + tau) / C_ij
```

The current global runs therefore genuinely test half-lives measured in global
option events.

The long/per-stream branch does not currently execute this fast-weight rule:
`hrl_persistent_fast_weights=False` activates the legacy best-time update
path. It does not decay weights, does not use the confidence threshold for
selection, and only marks `tctrl_updated` for an arrival that improves the
stored time. This invalidates its nominal 5k/10k/20k half-life comparison.

## Current Results in Context

Late-window values at this snapshot:

| Family | DG density | Silent DG fraction | Hit rate | Known edge fraction | Coverage AUC |
|---|---:|---:|---:|---:|---:|
| Fixed flat `encourage` | 0.0233 | 0.000 | n/a | n/a | 47.3 +/- 32.4 |
| Fixed/global HRL | 0.0405 | 0.0069 | 0.00249 | 0.152 | 58.9 +/- 4.6 |
| Long/per-stream HRL | 0.0474 | 0.0104 | 0.00243 | 0.186 | 48.3 +/- 16.7 |
| Long flat | 0.0356 | 0.0000 | n/a | n/a | 34.0 +/- 28.9 |

The most promising HRL result is fixed/global graph, not because its internal
distance is lower, but because it maintains DG activity, forms confidence-gated
graph edges, and has the best provisional coverage among HRL conditions. The
best nominal cell so far is 10k global option-event half-life with simultaneous
updates, AUC 61.5 across three seeds. This is a provisional candidate, not a
selected architecture, until the batch completes and seed-paired late windows
are compared.

## Metrics That Must Not Be Overinterpreted

- `distance_metric` is an internal temporal-distance statistic, not spatial
  coverage or an achievement score.
- Raw intrinsic reward is not comparable between flat and HRL runs because
  flat uses dense temporal reward while HRL uses a hit-gated reward.
- High known-edge fraction does not establish controllability; it must be
  accompanied by rising option success and coverage.
- Higher coverage AUC does not prove planning or target following. It is an
  external behavior outcome; target-hit and option-success trajectories are
  needed for that causal claim.
