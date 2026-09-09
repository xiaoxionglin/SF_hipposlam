# DG Structural Diversity

This iteration adds two default-off mechanisms to the existing trainable
`batchnorm_relu` DG projection. It does not change the frozen ImageNet
`layer2_resnet18` trunk, CA3 dynamics, decoder architecture, PPO targets, or
Sample Factory defaults. In policy-buffer HRL only, recruiting a DG row
invalidates graph evidence whose node identity has changed.

## CA3 temporal exclusion

The revised term is an event-level margin on the same dominant DG onset used by
the distance-weighted encoder objective. Let `D[t,j]` select that onset,
`d[t]` be its predecessor distance, `s=reward_scale`, and `c` be
`dg_ca3_temporal_exclusion_coeff`:

```text
a[t,j] = relu(z[t,j] - theta)
L_margin = c * s * R * mean_valid,t[sum_j D[t,j] a[t,j]]
```

For `encoder_reward_method=encourage`, the base term is
`-s * d[t] * a[t,j]`. At `c=1`, their sum is
`-s * (d[t] - R) * a[t,j]`: below-`R` onsets are suppressed, an onset at `R`
is neutral, and above-`R` onsets are reinforced. Non-dominant and continuing
activity is unaffected. Use `--dg_ca3_temporal_exclusion_coeff=0` to disable it;
nonzero use currently requires `encourage`.

The historical different-DG mask derived from CA3 slot `R-1` remains detached
and is still logged as conflict telemetry, but it no longer gates the loss.
Runs launched before 2026-09-03 used the legacy broad activity-masked loss and
must be treated as a different intervention.

## Orthogonal recruitment

Use `--dg_orthogonal_recruitment=True`. A candidate occurs when exactly one DG
pulse first enters the final CA3 slot, `L` decisions after it started, and no
different DG is present in the register. A same-source reactivation does not
restart the event. The first time step of a rollout is conservatively ignored
if the pulse was already at the tail because its previous-tail state is not in
that accepted rollout.

At a candidate endpoint, the learner recomputes the fixed visual/DG input after
PPO. If any DG is currently active, it skips the event. Otherwise it selects the
least-used row that has not previously been structurally recruited and computes
the component outside the row span of all other DG weights:

```text
B = orthonormal basis(row_span(W_without_j))
r = x - projection_B(x)
w_j = r / ||r||
```

Numerically tiny residuals are skipped. Assignments happen under the policy
lock, after PPO and before the normal actor synchronization. The selected
optimizer-moment row is zeroed. BatchNorm running variance is initialized from
the other rows, and running mean is set so the endpoint has normalized logit
`theta + margin`. Each row can be structurally recruited at most once, and
`--dg_orthogonal_recruitment_max_per_rollout` bounds accepted-rollout mutation
frequency. The default is one.

The per-rollout limit is an engineering rate limiter rather than part of the
learning rule. A learner batch contains many rollout streams and can expose
many silent endpoints at once; accepting one assignment prevents a single PPO
update from replacing a large fraction of the representation before actors
synchronize. Candidates later in that accepted rollout are intentionally
dropped. Across rollouts, recruitment can still reach the `F`-row lifetime
bound.

The committed mask, cumulative unit use, recruitment count, and tiny-residual
count are model buffers. They are included in checkpoints and policy state
synchronization but have no gradients. Old checkpoints load with zeroed
recruitment state.

For `hrl_graph_memory=policy_buffer`, row reassignment also clears that node's
visit weight and the corresponding rows and columns of `T_ctrl` and edge
confidence. It increments a model-buffer representation generation. Compact
actor option state stores the generation under which its target was selected.
After synchronization, a stale option cannot produce a hit, timeout, or graph
transition: it is reset while the target already stored for the current sampled
action remains replayable, and the next state receives a target selected from
the new graph. The learner rejects graph events and visit updates from late
rollouts carrying an older generation. Flat and episode-graph modes do not use
this field.

## Batch

`dg_structural_diversity.py` defines a 48-job, three-seed factorial:

```text
architecture: flat, fixed/global HRL
background: CTRL, global punishment 0.01 + row repulsion 1.0
CA3 exclusion: off, coefficient 1.0
orthogonal recruitment: off, on
seed: 8, 99, 123
```

This design estimates each mechanism, their interaction, compatibility with
the previous best regularizers, and whether representation changes depend on
HRL. All other settings match the previous best-candidate batch: fixed-length
no-reward DMLab, threshold 2.43, `F=16`, `R=8`, `L=64`, encourage plus batch
usage, simultaneous update, one policy, no PBT, 100M environment frames, and
fixed/global HRL `hit_distance` reward where enabled.

See `LOGGING.md` for metric definitions and the Obsidian experiment report for
preflight evidence and submitted Slurm job IDs.

Production was submitted on 2026-08-26 as Slurm jobs `7871544` through
`7871591`. The manifest is under the workspace `_slurm` tree for
`intrmotiv_dg_structural_diversity_20260826/20260826T121045Z`.
