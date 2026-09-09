# HRL Batch 1 Results and Next-Iteration Proposal

Date of analysis: 2026-08-19  
W&B project: `SF_HRL_Intrinsic_ArchSearch`  
Batch: `intrmotiv_hrl_batch1_20260818`  
Run family: `B1_F16_*`

## Executive conclusion

Batch 1 successfully exercised the recurrent HRL state, target conditioning,
episode-local controllability graph, and target-gated intrinsic reward at
scale. It did **not** demonstrate a working goal-directed exploration system.

The main result is a failure diagnosis rather than a winning configuration:

1. `encourage` is the only encoder feedback rule that reliably keeps the DG
   population alive. `mean` and `punish` leave about 73-76% of DG units silent
   in a typical learner minibatch.
2. An active target exists almost continuously, but exact target hits are very
   rare. Across all runs at 50-60M frames, only about 0.099% of transitions hit
   the target. Even `encourage` reaches only 0.253%.
3. Learned option deadlines are almost never selected: 0.47% overall and 1.40%
   for `encourage`. In 48 of 72 runs the measured fraction is exactly zero.
   Options therefore use the fallback `L` horizon almost all the time.
4. Worker reward is even sparser than target hits and is not guaranteed to be
   positive on success. A representative `encourage` run had mean intrinsic
   reward `-0.0050` at 50-60M frames, despite nonzero target hits, with sampled
   advantage-reward minima down to `-10.1`.
5. The sweep is confounded because the three encoder methods have radically
   different gradient scales, and the scale of `encourage` grows with `L`.
   The apparent advantage of `L=128` is therefore not evidence that longer CA3
   memory or longer options are intrinsically better.
6. The batch did not record a valid external exploration outcome such as
   spatial coverage. The logged `distance_metric` is an internal temporal
   matrix statistic, not environment coverage. Consequently, the batch cannot
   establish that any condition explores the map better.

The next iteration should not repeat the same 3 x 3 x 2 factorial sweep. It
should first correct reward semantics, target eligibility, activity control,
deadline estimation, episode length, and exploration measurement.

## Data coverage and method

The analysis reads TensorBoard event files directly from all 72 run
directories. This avoids dependence on W&B upload state.

At the analysis snapshot:

- 72/72 runs had readable event streams.
- All 72 reached at least 59.8M environment frames.
- Median maximum step was 90.03M frames.
- 54 reached 80M frames.
- 43 reached the 90M target and had completed.
- 2 had reached the Slurm time limit.
- 27 were still running and were not interrupted.

The main comparison uses the fixed 50-60M-frame window, available for every
run. Since `env_frameskip=8`, this 10M-frame window corresponds to about 1.25M
policy actions. Early 5-15M and mature 80-90M windows were also checked.

Statistics below are means of per-run window means. `dg_silent_unit_frac` is a
learner-minibatch statistic, not the fraction of units that are silent over an
entire training run.

Reproducible artifacts:

- Analyzer: `06_experiments/analyze_hrl_batch1.py`
- Per-run windows: `06_experiments/results/hrl_batch1_stats/per_run_windows.csv`
- Grouped CSVs: `06_experiments/results/hrl_batch1_stats/aggregate_*.csv`
- Batch summary: `06_experiments/results/hrl_batch1_stats/batch_summary.json`

## Overall HRL health

### All 72 runs at 50-60M frames

| Metric | Mean | SD | Minimum | Maximum |
|---|---:|---:|---:|---:|
| DG density per unit | 0.01153 | 0.01423 | 0.00032 | 0.04497 |
| Multi-activation rate | 0.01936 | 0.03151 | 0 | 0.13656 |
| Silent-unit fraction | 0.49773 | 0.36617 | 0 | 0.97477 |
| Active-target fraction | 0.99878 | 0.00008 | 0.99859 | 0.99894 |
| Target-hit rate per transition | 0.000993 | 0.001430 | 0 | 0.007424 |
| Option-success fraction | 0.06491 | 0.10607 | 0 | 0.50954 |
| Option-timeout rate per transition | 0.01749 | 0.00973 | 0.00572 | 0.03103 |
| Learned-deadline fraction | 0.004662 | 0.012927 | 0 | 0.08360 |
| Episode-local node coverage | 0.25412 | 0.22212 | 0.01302 | 0.83395 |
| Known-edge fraction | 0.03184 | 0.04489 | 0.00085 | 0.21267 |
| Intrinsic-reward mean | 0.003445 | 0.008227 | -0.00502 | 0.05203 |
| Nonzero intrinsic-reward fraction | 0.000527 | 0.000894 | 0 | 0.005067 |
| Policy entropy | 1.20419 | 0.41882 | 0.15287 | 1.59797 |
| Policy sample throughput | 277.3/s | 31.4/s | 200.9/s | 311.2/s |

Additional failure counts:

- 5/72 runs had zero target hits in the common window.
- 48/72 had zero learned-deadline selections.
- 46/72 had more than half of DG units silent per sampled minibatch.
- 7/72 had more than 90% silent.
- 31/72 produced nonzero worker reward on fewer than 0.01% of transitions.

The high active-target fraction therefore does not indicate successful HRL.
It only shows that a target ID is stored. The worker usually receives a target
that it does not reach and almost never receives a learning event for it.

## Encoder reward-method result

### Means at 50-60M frames, 24 runs per method

| Metric | `encourage` | `mean` | `punish` |
|---|---:|---:|---:|
| DG density per unit | **0.03039** | 0.00270 | 0.00151 |
| Silent-unit fraction | **0.00521** | 0.73086 | 0.75712 |
| Multi-activation rate | 0.05251 | 0.00352 | 0.00206 |
| Target-hit rate | **0.002531** | 0.000239 | 0.000211 |
| Option-success fraction | **0.16982** | 0.01257 | 0.01235 |
| Learned-deadline fraction | **0.01399** | 0 | 0 |
| Node coverage | **0.52951** | 0.12258 | 0.11026 |
| Known-edge fraction | **0.08065** | 0.00777 | 0.00711 |
| Nonzero reward fraction | **0.001383** | 0.000110 | 0.000086 |
| Mean episode length, engine frames | 6139 | 6615 | 6593 |

`encourage` is clearly less collapsed, but it is not yet good. Its target-hit
rate is still only one hit per roughly 395 transitions, and only about one in
six options succeeds. The 0.14% nonzero-reward fraction is too sparse for a
CPU PPO run with one epoch per batch.

The `mean` and `punish` variants are invalid as HRL experiments because most
of their possible subgoals do not occur. Their manager frequently targets DG
channels with zero prior visits. In the common window, their mean
`hrl_selected_target_visit_mean` is exactly zero.

### No meaningful improvement over training

The target-hit rates were already approximately established by 5-15M frames:

| Method | Hit rate at 5-15M | Hit rate at 50-60M | Option success at 5-15M | Option success at 50-60M |
|---|---:|---:|---:|---:|
| `encourage` | 0.002586 | 0.002531 | 0.16114 | 0.16982 |
| `mean` | 0.000211 | 0.000239 | 0.01258 | 0.01257 |
| `punish` | 0.000184 | 0.000211 | 0.01195 | 0.01235 |

There is no evidence of the worker progressively learning reliable target
reaching. The result is better explained by DG event frequency and random
encounters with selected targets.

## Threshold result

| Metric | Threshold 2.00 | Threshold 2.43 |
|---|---:|---:|
| DG density | **0.01352** | 0.00955 |
| Silent-unit fraction | **0.47251** | 0.52295 |
| Multi-activation rate | 0.02664 | **0.01209** |
| Target-hit rate | **0.001301** | 0.000686 |
| Option-success fraction | **0.08130** | 0.04852 |
| Learned-deadline fraction | **0.00773** | 0.00159 |
| Node coverage | **0.28816** | 0.22008 |

Threshold 2.00 is the better bootstrap setting. Threshold 2.43 reduces
multi-activation, but mainly by suppressing events and target opportunities.
The next iteration should use 2.00 and control collisions with an explicit
loss rather than suppressing the whole population with a higher threshold.

## Sequence length result and confound

| Metric | `L=32` | `L=64` | `L=128` |
|---|---:|---:|---:|
| Target-hit rate | 0.001126 | 0.000719 | 0.001135 |
| Option-success fraction | 0.03393 | 0.04380 | 0.11701 |
| Learned-deadline fraction | 0.00633 | 0.00166 | 0.00599 |
| Selected deadline | 31.92 | 63.95 | 127.58 |
| Timeout elapsed time | 31.91 | 63.95 | 127.33 |
| Sample throughput | **291.2/s** | 270.7/s | 270.0/s |
| Internal `distance_metric` | 2.45 | 3.53 | 8.75 |

The increase in option success at `L=128` is largely mechanical: an option is
given four times as long as at `L=32`. The per-transition target-hit rate does
not improve. Selected deadlines are almost exactly `L`, confirming that the
learned graph does not control the horizon.

The internal `distance_metric` also scales strongly with `L`; it is unsuitable
for comparing or selecting checkpoints across sequence lengths without
normalization.

### Encoder-gradient scale changes with `L`

For `encourage`, mean encoder feedback and loss were:

| `L` | Mean `rewards_encoder` | Mean encoder loss |
|---:|---:|---:|
| 32 | 3.62 | -2.34 |
| 64 | 6.93 | -6.30 |
| 128 | 12.31 | -10.58 |

For `mean` and `punish`, encoder losses were generally around `1e-4` to
`1e-3`. Thus the reward-method sweep compares signals differing by four or
five orders of magnitude, while the `L` sweep also changes feedback scale.
This must be corrected before interpreting either factor architecturally.

## Why the present architecture fails

### 1. Least-visited selection chooses dead or extremely rare DG units

`select_target_for_layout` scores every node only by visit count, excluding
the source and optionally the just-expired target. It has no minimum activity,
confidence, or feasibility criterion.

For collapsed representations, the lowest-count channels are precisely the
silent channels. The manager therefore repeatedly selects targets that cannot
be observed, let alone reached. Even under `encourage`, low-count targets are
rarer than ordinary DG events.

At 50-60M, `encourage` has approximately 0.486 active-unit events per
transition (`16 * dg_density`) but only 0.00253 exact target hits. The hit rate
is about 0.5% of the active-unit event rate, far below the 1/16 reference that
would result from uniformly matched targets and events.

This is not an argument for preferring cheap targets. Feasibility should be a
candidate mask; novelty should remain the selection objective among feasible
targets.

### 2. Graph updates and worker learning use different events

Every accidentally reached non-source DG can update `Tctrl[source, reached]`
when it improves the stored minimum. An exact target hit is not required.
Worker reward, however, is gated to the exact selected target.

This explains why `hrl_tctrl_update_rate` (0.01887 overall) is roughly 19
times the target-hit rate (0.000993). The graph accumulates passive shortcut
evidence while the worker receives almost no corresponding learning signal.
Calling all of these edges controllable overstates what was deliberately
achieved.

Qualified hindsight should update a predictive association or train a
segment-level auxiliary objective. It should not be treated as equivalent to
an intended successful option in the controllability estimate.

### 3. The graph does not affect target choice

`Tctrl` is consulted only after novelty-only target selection, to determine a
deadline. It neither supplies a feasibility mask nor helps choose a frontier.
Because least-visited targets generally have no known source-target edge, the
deadline lookup misses and falls back to `L`.

This produces the observed learned-deadline fraction of 0.47%. The graph is
maintained but is almost behaviorally inert.

### 4. Minimum successful time is not expected arrival time

`Tctrl[i,j]` stores the smallest observed elapsed time. The desired option
rule is based on expected arrival time plus a margin. A minimum is an
optimistic outlier estimator and becomes increasingly aggressive as samples
accumulate.

The state should retain at least a count and running mean; preferably mean and
dispersion or a hit-time quantile. A deadline can then be computed as

\[
H(i,j) = \lceil \mu_{ij} + \kappa\sigma_{ij} + m \rceil.
\]

Unknown-target fallback should be a separate bootstrap prior, not `L`. `L`
is CA3 memory length and should not simultaneously define option patience.

### 5. A target hit is not guaranteed to reward the worker

The worker reward is

\[
r_t = \mathbf{1}[g_t=j_t]\,(B-d_t)\,s,
\]

where `d_t` is the legacy temporal internal-reward quantity and
`B=L+R-1`. The gated temporal term was not designed as a target-achievement
reward. It can be zero or negative when the selected target is reached.

The run `B1_F16_L64_T200_ERencourage_S8` illustrates the problem at 50-60M:

- target-hit rate: 0.001747;
- mean intrinsic reward: -0.005022;
- sampled reward minimum averaged about -6.94;
- individual summary windows reached a minimum of -10.1.

The worker can therefore be punished for accomplishing its assigned option.
This destroys the semantic contract between manager target and worker reward.

### 6. The batch-usage loss does not balance population use

The current batch term only detects units absent from the entire current
minibatch and rewards their present activation. Once every unit appears once,
the loss is exactly zero. Under `encourage`, its common-window mean is zero;
under `mean` and `punish`, it is only about `-5e-4` and `-3e-4` while most units
remain silent.

It does not penalize highly unequal usage among non-silent units and has no
explicit target activity rate. A rolling usage-distribution objective is
needed, together with separate density and collision controls.

### 7. Episode termination confounds behavior and graph memory

The batch uses `openfield_map2_fixed_loc3_noreward`. Although goal reward is
zero, touching the goal still ends the episode. Mean episode length differs by
method (`encourage`: 6139 frames; `mean`: 6615; `punish`: 6593), so methods see
different graph-memory horizons and reset frequencies.

The registered replacement
`openfield_map2_fixed_loc3_fixedlength_noreward` disables goal termination and
runs every episode for 7,200 engine frames, or 900 actions at frameskip 8. It
should be mandatory for the next batch.

### 8. The batch lacks an exploration outcome

Environment return and `lenweighted_score` are zero by design. Episode length
is behavior dependent but is not a coverage metric. The internal temporal
`distance_metric`, DG node coverage, and graph edge coverage all depend on the
learned representation and can be increased by changing DG activity without
exploring more physical space.

No claim about curiosity or map exploration should be made from Batch 1. At
most, shorter episodes under `encourage` suggest more frequent contact with
the fixed goal; that is neither broad coverage nor a reliable objective.

## Corrected next-iteration architecture

### A. Make worker success unambiguously positive

Start with a direct option reward:

\[
r_t^{worker} = \mathbf{1}[g_t=j_t]
\left(1 + \lambda_d\,\operatorname{clip}(d_t/B,0,1)\right).
\]

Use `lambda_d=0` as the hit-only control and one small positive value as the
distance-scaled variant. This preserves the earlier idea that decoder updates
can depend on elapsed DG distance while guaranteeing that a target hit is
never punished.

Do not add a per-step time cost. Curiosity should not prefer cheap targets.
Timing belongs to feasibility and option switching, not the exploration
utility.

### B. Separate target eligibility from target novelty

Construct a candidate set before novelty ranking. A node is eligible when it
has sufficient activity evidence and either:

- a known intended-success edge from the current source; or
- sufficient predicted hit probability from the current CA3 state.

Within this set, choose the least visited node. Feasibility is a mask, not a
negative cost term. If no candidate is known, bootstrap from recently observed
non-source DG nodes rather than arbitrary silent channels.

Log candidate-set size, fraction of resets with no candidate, target activity
frequency, and hit rate conditional on any DG event.

### C. Correct controllability evidence and deadlines

Maintain separate evidence:

- intended target hits update controllable success count and hit-time moments;
- accidental DG arrivals update the predictive association or a separate
  observational statistic;
- failed options update attempt and timeout counts.

Store `(attempts, successes, mean_time, time_variance)` or equivalent. Use
expected time plus uncertainty for deadlines. The bootstrap horizon should be
a fixed prior measured in policy actions and should not vary with `L`.

### D. Normalize encoder feedback and explicitly control DG use

Normalize temporal feedback by `B=L+R-1` so that changing `L` does not change
gradient scale. Report both raw and normalized values.

Replace the current all-or-nothing batch term with three explicit terms:

\[
L_{DG} = L_{distance}
       + \lambda_{usage}D_{KL}(u\;||\;U_F)
       + \lambda_{density}(\bar a-\rho)^2
       + \lambda_{collision}L_{multi}.
\]

Here `u` is an EMA or sufficiently long-window DG usage distribution, `U_F`
is uniform over available units, `bar a` is mean activity, and `rho` is the
desired event density. The usage term should remain active when units are
unequally used, not disappear after one occurrence per minibatch.

For the next iteration, do not repeat the unnormalized `punish/encourage/mean`
sweep. Use normalized `encourage` and normalized centered feedback as the two
main variants; retain at most one `punish` control if scientifically needed.

### E. Add the full-CA3 predictor in shadow mode

Train the proposed association from the complete current CA3 state `S_t` and
target `j` to:

- probability of hitting `j` within a future window;
- expected or distributional hit time;
- optionally the next DG event identity.

In the next batch it should be logged but should not yet alter actions or
rewards. Measure calibration, hit-probability AUC, top-k next-DG accuracy, and
hit-time error. Once valid, it can provide:

1. the manager feasibility mask;
2. expected option deadlines;
3. potential-based worker shaping
   `gamma * Phi(S_{t+1},j) - Phi(S_t,j)`.

This predictor is persistent across episodes through learned weights and
addresses the principal limitation of an episode-local graph without adding a
second hand-written reachability network.

## Measurement required before PBT

### External exploration objective

Request DMLab position for telemetry but do not expose it in policy
observations. Record a multi-resolution spatial coverage AUC:

\[
J_{coverage} = \sum_s w_s\frac{1}{T}
\sum_{t=1}^{T}|\{\operatorname{cell}_s(x_\tau):\tau\le t\}|.
\]

Also log final unique-cell coverage and occupancy entropy. Coverage AUC is the
primary outcome because it rewards discovering new space early and cannot be
increased indefinitely by looping along the same path.

### Four-policy PBT

Use:

```text
--num_policies=4
--with_pbt=True
--policy_workers_per_policy=1
```

With `pbt_replace_fraction=0.2`, one worst policy can inherit from one best
policy per replacement round.

PBT donor ranking should be lexicographic:

1. HRL-valid policies only: noncollapsed DG activity, active candidate targets,
   and no systematic negative success rewards;
2. among valid policies, maximize external spatial coverage AUC.

Do not optimize intrinsic reward, target-hit rate, path length, graph density,
or minimum travel time directly. Each admits a failure mode unrelated to broad
exploration.

The stock PBT implementation reads `runner.policy_avg_stats`, which is fed by
episodic environment statistics. Current learner-only metrics such as
`distance_metric` are not automatically available there. A short preflight
must show both:

- a log message matching the intended PBT objective key; and
- a real save/load replacement from one policy to another.

Do not launch the full PBT batch until this test passes.

## Proposed next batch

### Fixed settings

```text
environment: openfield_map2_fixed_loc3_fixedlength_noreward
visual trunk: layer2_resnet18
F: 16
DG threshold: 2.0
L: 64 for the corrective iteration
encoder_batch_loss: replaced by rolling usage/density/collision controls
num_policies: 4
with_pbt: True
policy_workers_per_policy: 1
workers: benchmark 32 workers x 2 or 4 envs before production
worker CPU affinity: True unless the preflight disproves it
```

Use one middle `L` value while fixing semantics. Revisit `L` only after encoder
feedback is normalized and option deadlines are decoupled from it.

### Small factorial comparison

Use two encoder variants and two worker rewards:

| Factor | Variant 1 | Variant 2 |
|---|---|---|
| Encoder feedback | normalized encourage | normalized centered |
| Worker reward | positive target hit only | positive hit plus clipped normalized DG-distance bonus |

Run three independent population seeds per cell. Each cell then has three
independent four-policy PBT populations: 12 Slurm jobs and 48 policy learners.
PBT members are not independent statistical seeds, so population replication
is still necessary.

Add three population-seed runs of a non-HRL intrinsic baseline if resources
allow. This produces an external exploration reference and prevents accepting
an HRL mechanism that performs no better than flat intrinsic PPO.

Train the CA3 predictor in shadow mode in every corrected HRL run, using the
same architecture and logging, but do not create another predictor sweep yet.

### PBT mutations

Keep architecture and representation semantics fixed within each population.
PBT may perturb:

- policy learning rate;
- entropy coefficient;
- normalized worker distance-bonus coefficient;
- bounded encoder activity-control coefficients.

Do not mutate `F`, `L`, threshold, reward-method identity, or graph semantics
inside a population. Those changes make inherited optimizer/model state hard
to interpret and can be dimensionally incompatible.

## Acceptance criteria

The corrected HRL should proceed to a broader architecture sweep only if:

1. Every episode is exactly 7,200 engine frames / 900 policy actions.
2. No target hit produces negative worker reward.
3. DG silent-unit fraction remains below 25% in the evaluation window, and
   multi-activation remains controlled.
4. The conditional target-hit rate improves over training and exceeds matched
   random/non-HRL target assignment.
5. Learned/predicted deadlines are used on a substantial fraction of mature
   options; 25% is a reasonable initial threshold, compared with 0.47% now.
6. Spatial coverage AUC exceeds the non-HRL baseline across independent
   population seeds.
7. Predictor calibration and hit-time error improve over frequency/mean-time
   baselines before predictor shaping is enabled.
8. Four-policy PBT performs at least one verified inheritance event and does
   not select an HRL-invalid donor.

## What Batch 1 does establish

The implementation is capable of carrying a large episode-local graph in PPO
recurrent state and replaying target sequences without an obvious runtime
failure. Target conditioning, graph diagnostics, intrinsic reward summaries,
and all 72 experiment configurations ran long enough to diagnose behavior.

The useful hyperparameter conclusions are limited but clear:

- threshold 2.0 is a better bootstrap point than 2.43;
- unnormalized `punish` and `mean` collapse too much of the DG population;
- `encourage` is a viable activity bootstrap but needs scale normalization;
- option success at large `L` is not evidence of learned timing;
- the next scientific bottleneck is objective and control semantics, not a
  larger factorial sweep.
