# IntrMotiv Metric Logging

IntrMotiv-specific learner metrics are written under `intrmotiv/` in
TensorBoard and W&B. They are removed from the learner message before Sample
Factory's generic handler runs, so they no longer appear as duplicate
`train/...` series. Generic PPO and framework metrics remain under `train/`.

## Metric Groups

| Group | Metrics | Meaning |
| --- | --- | --- |
| `intrmotiv/distance/` | `mean`, `min`, `max`, `std`, `masked_*` | Pairwise CA3 progression-distance diagnostics. These are observational metrics, not the no-PBT batch objective. |
| `intrmotiv/update/phase` | `0`, `1`, `2` | `0`: simultaneous, `1`: decoder-only, `2`: DG-projection encoder-only. |
| `intrmotiv/reward/` | `advantage_*`, `intrinsic_*`, `environment_*` | PPO learning reward, transition-distance intrinsic reward, and original environment reward. Fractions are in `[0, 1]`. |
| `intrmotiv/dg/` | `density`, `multi_activation_fraction`, `silent_unit_fraction`, `behavior_*`, `valid_minibatch_unused_unit_*`, `active_count*` | Learner activity plus corrected behavior-onset and recruitment diagnostics. |
| `intrmotiv/encoder/` | `loss`, `feedback_mean`, `feedback_*_on_dominant_event_mean`, `dominant_event_count`, `batch_usage_loss`, `unused_sequence_loss`, `multi_activation_loss`, population terms | Encoder objective, event-conditional feedback, and enabled auxiliary terms. Disabled terms are logged as zero. |
| `intrmotiv/decoder/` | `loss`, `auxiliary_loss` | Decoder-side optimization terms. |
| `intrmotiv/predictor/` | `loss`, `hit_accuracy`, `hit_time_mae`, `positive_fraction` | CA3 target-predictor diagnostics when that optional module is enabled. |
| `intrmotiv/hrl/` | target, option, deadline, node, `tctrl`, and fast-weight statistics | Controllable-graph diagnostics, present only when HRL is enabled. |
| `intrmotiv/pbt/` | `hrl_validity`, `objective` | PBT-only routing diagnostics. `distance_metric` is intentionally excluded because it is not an exploration objective. |
| `intrmotiv/exploration/window/` | `coverage_*`, `pose_*` | Fixed-window external exploration metrics. `coverage_*` bins `(x, y)`; `pose_*` additionally bins DMLab yaw. The private pose is removed in the environment wrapper and never enters policy observations, rewards, or model inputs. |
| `intrmotiv/online/place_field/` | sample/bounds/occupancy fractions, active/silent fractions, spatial information, active-map cosine, peak bins | Scalar-only latest-10k behavior-time place-field monitoring on the standard 19×19 grid. |
| `intrmotiv/online/trajectory/` | step distance, stationary fraction, path efficiency, circular yaw change | Terminal- and rollout-segment-aware latest-10k trajectory monitoring. |

## Compact Online Spatial Telemetry

`--online_spatial_telemetry=True` is default-on and independently disableable.
DMLab declares a float32 `telemetry_pose=(x,y,yaw)` rollout channel, but the
model is constructed from an observation space without that channel and
`prepare_and_normalize_obs` removes it before device conversion and
normalization. The actor exports its exact behavior-time thresholded DG head
as a detached policy output. The learner joins those arrays only after
policy-ID and policy-lag filtering.

W&B receives the grouped scalar namespaces above every 1M environment frames;
there are no training-time `wandb.Image` calls. Per-policy compressed snapshots
are written at 25M, 50M, 75M, and 100M under
`<train_dir>/analysis/online_spatial/<experiment>/policy_NN/`. The v1 artifact
contains pose, thresholded DG activity, actions, dones, segment IDs, policy
versions, target/actual frame counts, window limits, identity, frameskip,
bounds, and schema metadata. It excludes RGB, recurrent state, pre-threshold
logits, and parameters. Writes are atomic and an existing valid target is kept.
After resume, cadence starts strictly after restored environment frames.

Use the standardized workflow's `collect-spatial` command for CSV aggregation
and explicitly selected post-hoc figures. These training windows are
diagnostics; controlled checkpoint rollouts remain authoritative.

## Corrected DG Event Diagnostics

These metrics separate stored behavior-time DG onset labels from the learner's
current post-threshold activity. They use only valid learner transitions.

| Metric | Definition |
| --- | --- |
| `intrmotiv/dg/learner_active_transition_fraction` | Valid transitions with at least one active DG in the current learner forward. This includes sustained fields and is not itself an onset rate. |
| `intrmotiv/dg/behavior_dominant_event_fraction` | Valid transitions carrying one stored dominant behavior onset. |
| `intrmotiv/dg/behavior_multi_onset_event_fraction` | Dominant behavior events that also had at least one simultaneous non-dominant candidate. |
| `intrmotiv/dg/behavior_non_dominant_onsets_per_event` | Number of stored non-dominant candidates divided by dominant behavior events. |
| `intrmotiv/dg/valid_minibatch_unused_unit_count` | DG rows absent from incoming CA3 and current activity across every valid minibatch transition. |
| `intrmotiv/dg/valid_minibatch_unused_unit_fraction` | The same count divided by `F`. |
| `intrmotiv/encoder/dominant_event_count` | Dominant behavior events in the summarized minibatch. |
| `intrmotiv/encoder/feedback_on_dominant_event_mean` | Mean signed encoder feedback over dominant behavior-event transitions. |
| `intrmotiv/encoder/feedback_abs_on_dominant_event_mean` | Mean absolute encoder feedback over those transitions. |

The learner-active and behavior-dominant fractions have different semantics;
their numerical difference is not a direct BatchNorm replay-error estimate.

## Topological Frontier Diagnostics

These metrics are emitted by `topology_visit_direct`, `frontier_direct`, or
`frontier_waypoint`.
Event rates use valid learner transitions; `*_per_rollout` values are counts
from the most recently accepted rollout. Graph fractions use all directed
off-diagonal DG pairs.

| Namespace | Primary metrics | Interpretation |
| --- | --- | --- |
| `intrmotiv/hrl/passive/` | `updates_per_rollout`, `known_edge_fraction`, `candidate_edge_fraction`, `traversal_time_mean`, `path_length_mean`, `reject_*_rate` | Whether exclusive local DG transitions are being accepted, and which gate rejects them. A candidate is observed at least twice but is not yet controllable. |
| `intrmotiv/hrl/frontier/` | `score_mean`, `selection_rate`, `attempts_per_rollout`, `discoveries_per_rollout`, `yield`, `reached_fraction` | Whether UCB curiosity creates productive exploration from reachable landmarks. Yield is decayed discoveries divided by decayed attempts. |
| `intrmotiv/hrl/planning/` | `route_available_rate`, `hop_count_mean`, `target_hit_rate`, `waypoint_navigation_fraction`, `waypoint_step_hit_rate`, `replan_rate`, `final_frontier_reach_rate` | Whether the validated graph supports routes, how often genuine multi-hop navigation is active, and its local hit signal. |
| `intrmotiv/hrl/validation/` | `queued_edges`, `return_success_rate`, `success_rate`, `timeout_rate` | Health of passive-edge deliberate validation. Passive edges never enter planning before success. |
| `intrmotiv/path/` | `path_length_mean`, `displacement_mean`, `straightness_mean`, `scatter_conflict_fraction`, `scatter_loss`, `telemetry_error` | Repeated-command path integration and same-DG far-reactivation pressure. Turning commands retain their full repeated path, not chord length. `telemetry_error` is episode-trajectory RMSE after similarity alignment to DMLab debug positions. |
| `intrmotiv/geometry/` | `se2_stress`, `valid_landmark_fraction`, `proposed_edge_fraction` | Pose-graph fit and geometry-only proposal coverage for the matched SE(2) control; the proposed-edge fraction is zero when geometry is disabled. |
| `intrmotiv/hrl/target_hit_lift` | scalar ratio | Current-target activation rate divided by a within-minibatch shifted-target activation baseline. Values above one indicate target-specific behavior beyond target frequency. |

Use these checks in order. First require viable DG density and low silent-unit
fraction. Next require passive updates and candidate edges. Then require
validation successes and a growing deliberate known-edge fraction. Only after
those prerequisites interpret route availability, multi-hop count, final
frontier reach, target-hit lift, and environment coverage. A high coverage AUC
without graph/validation activity is behavioral exploration, not evidence that
the topological algorithm is operating as intended.

## Flat Iterative Batch

For `SF_IntrMotiv_FlatBaseline_Iterative`, use these primary panels:

1. `intrmotiv/reward/intrinsic_mean` and `intrmotiv/reward/advantage_mean`
   verify that PPO is trained on the transition-distance reward.
2. `intrmotiv/dg/density`, `intrmotiv/dg/silent_unit_fraction`, and
   `intrmotiv/dg/multi_activation_fraction` diagnose landmark activity.
3. `intrmotiv/update/phase`, `intrmotiv/encoder/loss`, and
   `intrmotiv/decoder/loss` compare simultaneous with iterative learning.
4. DMLab coverage and occupancy metrics remain environment statistics outside
   this namespace and are the external exploration evaluation.

## Persistent HRL

Persistent HRL preserves only a per-rollout-stream graph suffix across a DMLab
terminal reset. CA3 state, active option, source, age, and deadline reset at
the boundary, so a respawn is never treated as a physical transition. The
preserved suffix contains node visit weights, `T_ctrl`, and edge confidence.

At each option reset, visit and edge confidence weights decay by:

```text
gamma = 0.5 ** (1 / hrl_fast_weight_half_life_options)
```

For a deliberate successful transition `i -> j` in `tau` actions, confidence
increments by one and `T_ctrl[i,j]` becomes the confidence-weighted mean of
the old estimate and `tau`. Edges below `hrl_edge_confidence_threshold` are
forgotten: they are not feasible for multi-hop planning or learned deadlines.

Use these panels for the persistent-HRL half-life sweep:

1. `intrmotiv/hrl/node_visit_weight_mean`,
   `intrmotiv/hrl/edge_confidence_mean`, and
   `intrmotiv/hrl/forgotten_edge_fraction` verify the fast-weight timescale.
2. `intrmotiv/hrl/known_edge_fraction` and
   `intrmotiv/hrl/known_controllability_time_mean` measure usable graph
   structure.
3. `intrmotiv/hrl/target_hit_rate`, `option_success_fraction`, and
   `intrmotiv/reward/intrinsic_mean` connect the graph to worker behavior.

`intrmotiv/hrl/selected_deadline_positive_mean` averages only strictly positive
deadlines, i.e. resets where target selection produced an option. Use this
instead of the compatibility metric `selected_deadline_mean`, whose denominator
includes all resets. Targetless resets are reported separately through
`intrmotiv/hrl/deadline_selection_fraction`, the fraction of option resets with
a positive selected deadline. The legacy metric is retained unchanged so
existing and resumed W&B series remain comparable.

## Persistence Comparison

`hrl_graph_memory=policy_buffer` keeps one non-gradient graph per policy in
the model state dict. The learner updates it once for each accepted rollout;
actors receive the next synchronized snapshot. The target that conditioned an
action is retained in that action's compact RNN option state and is supplied
unchanged during PPO replay. Therefore graph updates never alter replayed
worker targets.

For this mode the half-life is measured in **global** option completion or
timeout events. After each event, node visits and edge confidence decay by
`0.5 ** (1 / h)`. A successful `i -> j` event at elapsed time `tau` then
updates the confidence-weighted arrival-time estimate:

```text
C[i,j] <- gamma * C[i,j] + 1
T_ctrl[i,j] <- (gamma * C_old[i,j] * T_ctrl[i,j] + tau) / C[i,j]
```

`hrl_graph_memory=episode` retains the complete graph in each stream's normal
RNN state. The long-episode environment has a 36,000-second DMLab timeout, so
it does not terminate during a 100M-frame job; this mode does not use the
persistent-RNN suffix.

Long episodes emit `intrmotiv/exploration/window/` metrics every 900 policy
actions without declaring an environment terminal. Use window return, length,
coverage AUC, unique cells, and occupancy entropy for long-run evaluation.
They are telemetry windows, not physical episodes and do not reset CA3, an
option, or the graph.

## Manager Exploration Option

With `hrl_exploration_mode=True`, the fixed manager can select a reserved
exploration option at each option boundary with probability
`hrl_manager_exploration_probability`. A DG-target timeout always forces the
next option to exploration, including when that probability is zero. The
exploration option lasts `hrl_exploration_horizon` decisions and then returns
to ordinary novelty-first DG-target selection.

The reserved option is stored as target ID `F` in compact RNN option state. It
is supplied to the worker as the existing all-zero target vector, so no policy
input dimension changes. PPO replay teacher-forces the stored manager action;
it never resamples the exploration decision from the learner's newer graph.
During exploration, the worker receives the flat decoder's dense temporal-
distance reward. During a DG-target option it continues to receive the sparse
`hit_distance` reward. The feature is restricted to
`hrl_graph_memory=policy_buffer`, where stored-target replay is available.

Exploration completions are encoded internally with negative completion
elapsed time so target and exploration expirations remain distinguishable
without changing the recurrent-state layout. Reported elapsed metrics are
positive magnitudes.

| Metric | Meaning |
| --- | --- |
| `intrmotiv/hrl/active_option_fraction` | Fraction of transitions with either a DG target or exploration manager action. |
| `intrmotiv/hrl/exploration/mode_fraction` | Fraction of transitions executed under exploration. |
| `intrmotiv/hrl/exploration/selection_fraction` | Exploration selections divided by all option resets. |
| `intrmotiv/hrl/exploration/forced_selection_fraction` | Timeout-forced selections divided by all exploration selections. |
| `intrmotiv/hrl/exploration/completion_rate` | Exploration horizon completions per valid transition. |
| `intrmotiv/hrl/exploration/elapsed_mean` | Mean completed exploration duration. |
| `intrmotiv/hrl/exploration/selected_deadline_mean` | Mean configured deadline over exploration selections. |
| `intrmotiv/hrl/exploration/reward_mean` | Mean dense worker reward on exploration transitions. |
| `intrmotiv/hrl/exploration/reward_nonzero_fraction` | Nonzero dense worker rewards divided by exploration transitions. |
| `intrmotiv/hrl/target_selected_deadline_mean` | Mean deadline over newly selected DG-target options only. |

`option_success_fraction` remains target-only: hits divided by target hits plus
target timeouts. Exploration horizon completions do not lower it.

## DG Anti-Collapse Controls

The DG anti-collapse experiment intentionally keeps every batch-wise
population, usage, density, collision, and multi-activation objective disabled.
It instead evaluates two separate, non-batch-wise mechanisms:

1. **Global pre-threshold punishment.** For each valid transition and every DG
   unit, the learner penalizes the BatchNorm logit `z` before the hard DG
   threshold `theta` using:

   ```text
   L_global = lambda * mean[T * softplus((z - theta) / T)]
   ```

   The smooth surrogate gives inactive units a nonzero gradient. Because it is
   data-dependent and DG rows are normalized after optimization, it can rotate
   rows away from frequently visited visual features. `T` is
   `dg_global_punishment_temperature` and `lambda` is
   `dg_global_punishment_coeff`.

2. **Angular row repulsion.** This separate arm operates only on DG projection
   weights and never on a rollout batch:

   ```text
   L_row = lambda_row * mean_{i != j}[(normalize(w_i)^T normalize(w_j))^2]
   ```

   It prevents duplicate DG directions without explicitly suppressing
   activations. It is enabled by `dg_row_repulsion_coeff`.

The following panels diagnose these arms:

| Metric | Meaning |
| --- | --- |
| `intrmotiv/encoder/global_punishment_loss` | Weighted pre-threshold penalty included in the encoder objective. |
| `intrmotiv/encoder/row_repulsion_loss` | Weighted off-diagonal squared-cosine penalty. |
| `intrmotiv/dg/pre_threshold_mean` | Mean BatchNorm DG logit before thresholding. |
| `intrmotiv/dg/pre_threshold_above_fraction` | Fraction of DG logits above the hard threshold before ReLU. |
| `intrmotiv/dg/density` | Post-threshold DG density; compare this with the pre-threshold fraction. |
| `intrmotiv/dg/silent_unit_fraction` | Fraction of DG rows never activated in the learner window. |

For controls with both coefficients zero, the two loss metrics are zero but
the pre-threshold diagnostic metrics remain populated. This permits direct
comparison of threshold-only and penalty conditions.

## DG Structural Diversity

CA3 temporal exclusion is now an event-level margin aligned with the dominant
DG-onset encoder reward. Let `D[t,j]` select the dominant onset, `d[t]` be its
predecessor distance, `s=reward_scale`, and `c` be the configured coefficient.
The added loss is:

```text
a[t,j] = relu(z[t,j] - theta)
L_margin = c * s * R * mean_valid,t[sum_j D[t,j] * a[t,j]]
```

With `encoder_reward_method=encourage` and `c=1`, the combined event loss is
`-s * (d[t] - R) * a[t,j]`: distances below `R` are suppressed, distance `R`
is neutral, and distances above `R` are reinforced. Non-dominant and continuing
activity is not penalized. Zero disables the term, and nonzero use requires
`encourage` feedback. The historical different-DG CA3 mask remains available
for conflict diagnostics but no longer gates the loss. Runs launched before
2026-09-03 used the legacy broad conflict-masked activity penalty.

Orthogonal recruitment is learner-owned structural plasticity. A candidate is
emitted once when a lone source DG pulse first reaches the final CA3 slot,
exactly `L` policy decisions after the pulse began. Reactivation of that same
source does not restart the timer; any different DG in the register rejects the
candidate. If no existing DG is active at the endpoint, the least-used row that
has not previously been structurally recruited is replaced after PPO finishes
the accepted rollout:

```text
W_other = all DG rows except selected row j
B = orthonormal basis(row_span(W_other))
r = x - B^T B x
w_j <- r / ||r||
```

The update is skipped for a numerically small residual. A row can be recruited
at most once, and `dg_orthogonal_recruitment_max_per_rollout` additionally
limits mutation frequency. The learner resets optimizer moments for row `j`
and calibrates its BatchNorm running mean so the assignment observation has
pre-threshold value `theta + dg_orthogonal_recruitment_margin`. Recruitment
buffers are checkpointed and synchronized with the normal policy model. Old
checkpoints load with empty recruitment state.

The first transition in each accepted rollout cannot prove that a pulse entered
the CA3 tail during that rollout, so a tail already present there is suppressed.
This avoids counting a long-lived tail pulse again at every 64-step learner
boundary; an entry exactly across a boundary can be missed. In policy-buffer
HRL, recruiting row `j` clears node visit `j`, row/column `j` of `T_ctrl`, and
row/column `j` of edge confidence. A checkpointed representation generation in
the model and compact actor option state resets stale options and rejects stale
rollout graph updates after actor synchronization.

| Metric | Meaning |
| --- | --- |
| `intrmotiv/encoder/ca3_temporal_exclusion_loss` | Weighted active-only temporal exclusion loss. |
| `intrmotiv/dg/ca3_conflict_fraction` | Fraction of unit-transition entries masked by another recent DG; potential-conflict mask coverage, retained for compatibility. |
| `intrmotiv/dg/ca3_conflicting_activation_fraction` | Fraction of current post-threshold DG activation entries whose unit is masked by another DG active in the preceding R decisions. This is the primary exclusion-violation rate. |
| `intrmotiv/dg/ca3_conflict_activity` | Mean current DG activity among masked entries. |
| `intrmotiv/dg/recruitment/candidate_count` | L-step CA3 candidate events in the most recently reported rollout. |
| `intrmotiv/dg/recruitment/silent_endpoint_count` | Candidate endpoints with no current DG activation. |
| `intrmotiv/dg/recruitment/rollout_count` | Rows recruited from the most recently reported rollout. |
| `intrmotiv/dg/recruitment/total` | Cumulative structurally recruited rows; bounded by `F`. |
| `intrmotiv/dg/recruitment/committed_fraction` | Fraction of rows already structurally recruited. |
| `intrmotiv/dg/recruitment/residual_norm` | Norm after unit normalization; one for a successful update. |
| `intrmotiv/dg/recruitment/tiny_residual_total` | Cumulative candidates skipped because the residual was too small. |

## Goal Conditioning And Empirical HER

| Metric | Meaning |
| --- | --- |
| `intrmotiv/hrl/goal_condition/target_valid_fraction` | Fraction of worker decisions with a valid stored behavior target. |
| `intrmotiv/hrl/goal_condition/action_sensitivity` | Mean action-logit change after counterfactually shuffling stored targets within a learner minibatch. Near zero indicates the decoder is ignoring target input. |
| `intrmotiv/hrl/goal_condition/value_span` | Mean critic-value change under the same shuffled target control. |
| `intrmotiv/her/accepted_segments_per_rollout` | Number of same-episode DG hindsight segments accepted per rollout stream. |
| `intrmotiv/her/segment_length` | Mean number of relabeled decisions in an accepted segment. |
| `intrmotiv/her/positive_fraction` | Fraction of ordinary rollout decisions participating in a hindsight segment. |
| `intrmotiv/her/terminal_reward` | Mean terminal `hit_distance` reward used by the hindsight return. |
| `intrmotiv/her/behavior_logprob_ratio` | Mean `pi(a|s,g_hindsight) / pi_old(a|s,g_behavior)` ratio; this is diagnostic, not an unbiased importance correction. |
| `intrmotiv/her/clip_fraction` | Fraction of hindsight ratios outside PPO's clipping interval. |
| `intrmotiv/her/loss`, `policy_loss`, `value_loss` | Weighted auxiliary, actor, and critic contributions from empirical PPO-HER. |
| `intrmotiv/her/skipped_no_endpoint_per_rollout` | Streams with no valid distinct exclusive DG future endpoint. |
