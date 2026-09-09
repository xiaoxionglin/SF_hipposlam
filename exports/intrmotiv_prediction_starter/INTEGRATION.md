# Sample Factory integration contract

The student repository was not available for inspection. These are tensor and
update contracts, not verified line-by-line patches against her Sample Factory
version. Keep changes inside her existing custom model/learner and environment
wrapper where possible. Do not replace Sample Factory's learner with ours.

## 1. Establish one timeline

At decision t:

1. Receive observation o_t, containing the previously executed action a_(t-1).
2. Encode real DG d_t and advance real CA3 to h_t.
3. The actor reads its usual inputs and samples a_t.
4. Execute a_t, producing reward r_t, done_t, and the next observation.

The predictor's starting state is **h_t after processing o_t**, paired with
candidate a_t. A returned recurrent state from processing o_t is not the state
before o_t; advancing it again would double-count the observation.

The attached `prev_action` must be the action actually sent to the environment,
including any wrapper override. On reset use sentinel A (number of actions),
encode it as all zeros, and clear history and recurrent state. Insert past
actions chronologically. Never insert a hypothetical candidate into real state.

For the shadow experiment, previous-action plumbing is optional: the exported
head itself only takes CA3 and a prediction query. Add history only when the
student's model actually consumes it; expand declared state sizes consistently.

## 2. Add the head and a separate optimizer

Register `CA3TargetPredictor(ca3_size, n_dg, hidden_size)` as a model submodule so
its weights can be checkpointed. Give it a separate optimizer and save/restore
that optimizer too. Exclude its parameters from the PPO optimizer. Do not change
the actor/critic input widths in this first experiment.

Only the learner needs this head initially. If actor copies also instantiate
it, normal model synchronization can carry its weights; its output is unused.

## 3. Obtain aligned temporary training tensors

Use existing rollout batches or their chronological recurrent reconstruction:

| Tensor | Shape | Meaning |
|---|---|---|
| CA3 | `[B,T,F*E]` | Register after each real observation; exclude HRL metadata and bypass channels |
| DG | `[B,T,F]` | Actual nonnegative detector output for the same observations |
| Query | `[B,T,F]` | One-hot target whose future appearance is being predicted |
| Dones | `[B,T]` | Action at t ends the episode before observation t+1 |
| Valids | `[B,T]` | Excludes padding, rejected samples, and unavailable observations |

Construct labels before shuffling samples. Do not infer chronological order by
reshaping already shuffled or variable-length PackedSequence data. Use the
framework's trajectory/packing metadata and compare stepwise versus replayed
CA3 before training. Stored CA3, replayed DG, and policy snapshot must refer to
compatible representation versions.

The supplied query-conditioned head is a diagnostic scaffold. Queries may be
sampled independently of future outcomes, or all DG targets can be evaluated.
Do not choose only targets that later activate: this removes negatives.

`future_target_labels(query, dg, dones_after, H, valids)` uses the full requested
H. It masks unresolved windows at rollout/reset/padding boundaries. An observed
positive is usable even if the remainder of the window is unavailable.

This helper deliberately does not stitch batches. For H near or beyond rollout
length, report usable/censored fractions. The next development step is bounded
pending prediction records keyed by environment/episode/policy version, rather
than interpreting missing futures as failures. Discard or censor at resets;
retain only the information needed for supervision. No permanent transition
database is necessary.

## 4. Run a head-only update

```python
hit, delay, usable = future_target_labels(query, dg, dones_after, H, valids)
loss, metrics = shadow_prediction_loss(
    predictor,
    ca3.reshape(-1, ca3_size),
    query.reshape(-1, n_dg),
    hit.flatten(), delay.flatten(), usable.flatten(), H,
)
predictor_optimizer.zero_grad()
loss.backward()
predictor_optimizer.step()
```

The helper detaches CA3, queries, and labels. No predictor gradient reaches DG,
the visual encoder, or the actor. The existing PPO update remains separate and
uses only real actions with their actual stored behavior log-probabilities.
For the clean first experiment, use a frozen baseline policy and DG detector.

Freeze **normalization statistics as well as parameters** for that baseline.
Global `.train()` calls can reactivate BatchNorm updates: explicitly keep the
frozen detector in evaluation mode. Verify actor-side and learner-side DG
outputs agree for the same observation/history. No-grad alone does not freeze
BatchNorm buffers.

## 5. Check the integration before interpreting accuracy

- Predictor-off reproduces original actor outputs and recurrent trajectories.
- Head-only updates leave actor outputs, DG weights, and DG normalization
  buffers unchanged on a fixed observation sequence.
- Previous action is aligned with the observation it produced, including reset.
- Stepwise sampling and learner recurrent replay agree, including packed lengths.
- No cross-episode labels; truncated horizons are counted as censored.
- Log positive fraction and a majority/frequency baseline next to accuracy.
- Report time MAE in **policy decisions**, on observed positives only; multiply
  by action repeat only when reporting engine frames.
- Evaluate on held-out episodes/trajectories, not overlapping windows from the
  same trajectory. The upstream contextual classifier's event split is only an
  internal diagnostic, not a strong held-out generalization test.

## 6. Leave policy improvement for the next stage

After the action-conditioned event model is validated, imagined returns can
supervise the actor through an explicit auxiliary objective. No such objective
is included here. Define continuation behavior, current reward, discounting,
termination, and policy-version validity before adding it. Changing the PPO
policy changes a model defined as “first action, then follow PPO.”

Do not splice imagined latent states into real `rnn_states`, fabricate ordinary
PPO samples, or bootstrap a relocated-reward experiment exclusively from the
unchanged old-task critic.
