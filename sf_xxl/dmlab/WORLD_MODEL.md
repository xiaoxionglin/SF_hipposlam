# Enable and edit the DG world model

This is the integrated starting point for the action-input / planning project.
It trains a **shadow predictor alongside ordinary PPO**, with its own optimizer.
It does not yet perform planning or alter action selection.

## Enable it

Append this to the existing working HippoSLAM training command:

```bash
--dg_world_model=true
```

Supported launch modules:

- `python -m sf_xxl.dmlab.train_hipposlam`
- `python -m sf_working_directories.default.dmlab.train_hipposlam`

Keep the rest of the student's known-good environment, backbone/checkpoint, DG,
and recurrent-core arguments. This is not wired into the separate IntrMotiv HRL
launcher. The paper launcher currently imports its core/encoder from
`sf_working_directories/default`; that pre-existing routing is preserved.

Optional settings:

```bash
--dg_world_model_horizon=16 --dg_world_model_hidden_size=128 --dg_world_model_lr=0.001
```

Use a continuous DG shift-register core such as `--core_name=BypassSS`, a shared
recurrent actor/critic, and a single discrete action space. Recurrence must be
at least two. Start with horizon smaller than recurrence so fully observed
negative examples exist. The existing `--dmlab_reduced_action_set=true` is
appropriate if that is already the baseline action space.

The flag defaults to false and returns the original DefaultLearner. Actor
architecture, state size, sampling, PPO losses, optimizer, and checkpoints'
`model`/`optimizer` fields are unchanged even when prediction is enabled.

## Where to modify code

| File/function | Student's edit point |
|---|---|
| `world_model.py:DGEventWorldModel` | CA3 + candidate-action inputs and prediction architecture |
| `world_model.py:next_dg_event_labels` | Event definition, horizon, masks and future labels |
| `world_model.py:event_prediction_loss` | Auxiliary prediction objective |
| `world_model_learner.py:_forward_pass` | Capture detached CA3/DG and executed actions from the same PPO forward |
| `world_model_learner.py:_after_optimizer_step` | One separate model update after each real PPO minibatch update |
| `world_model_learner.py:_get_checkpoint_dict/_load_state` | Save/resume head and optimizer |

There are no changes inside `sample_factory/` and no dependency on the exported
starter-kit package. The extension reuses the existing learner factory and
post-optimizer hook.

## Current model and training cycle

1. The actor executes actions normally; real observations update real CA3.
2. During PPO replay, the learner captures the CA3 prefix **after the current
   observation**, the corresponding DG output, and the executed action a_t.
   The capture is detached and does not rerun the encoder or update BatchNorm a
   second time. The restored recurrence order, done mask, and valid mask are used.
3. Labels identify the first future timestep t+delta with any positive DG input,
   for delta in 1..H. Multiple positive DG channels and amplitudes are retained.
4. The standard PPO update proceeds unchanged.
5. The head alone learns occurrence within H, the conditional DG activation
   vector/amplitudes, and conditional arrival time. Metrics enter normal summaries.

Predictions mean **take the supplied first action, then follow the behavior
represented in the rollout**. They are not action-independent reachability and
do not specify arbitrary future action sequences. Changing the policy changes
the prediction target. Only the actually executed branch is supervised.

The same DG channel remaining active on the next step counts as an event.
This makes the skipped interval strictly DG-silent. It differs from IntrMotiv's
first-distinct-landmark labels. A predicted discrete vector and arrival time
could later drive copied CA3 dynamics; current outputs are point estimates and
independent activation probabilities, not a validated stochastic rollout model.

Prediction gradients never reach the actor, critic, encoder, or real recurrent
state. The learner-only head is not in PPO's optimizer or gradient clipping, and
its initialization preserves the actor's random-number state. In asynchronous
runs, additional compute can still change throughput and sampling order; we do
not claim full-run trajectory identity across schedules.

## Metrics

Normal TensorBoard/W&B training summaries include:

```text
train/dg_world_model/loss
train/dg_world_model/usable_count
train/dg_world_model/usable_fraction
train/dg_world_model/positive_count
train/dg_world_model/positive_fraction
train/dg_world_model/hit_accuracy
train/dg_world_model/dg_exact_match
train/dg_world_model/time_mae_decisions
```

Time error and DG exact match are conditional on observed positive events.
Their zero value when positive_count is zero means no eligible examples, not
perfect prediction. These are online training diagnostics, not held-out metrics.
Compare hit accuracy with the class-frequency baseline and inspect sparse DG
activity before interpreting a high score. Validate on held-out trajectories
before using predictions to train or select actions.

Incomplete horizons at recurrence boundaries, resets, or invalid samples are
**censored, not negative examples**. Positive observations can still be used.
There is no cross-batch pending-event collector in this first version. H >=
recurrence can yield positives only and is unsuitable for calibrated occurrence
prediction; a warning and usable_fraction expose this limitation.

## Checkpoints and prediction inspection

Normal checkpoints contain an additional top-level `dg_world_model` entry with
schema, dimensions/horizon, weights and optimizer state. Full training resume
restores both PPO and the head. An older PPO-only checkpoint initializes a fresh
head; loading an existing head with changed dimensions/horizon fails explicitly.
Use a new experiment for such changes. Preserve the world-model flag when
resuming; SF's saved experiment configuration may otherwise override CLI values.

For trusted checkpoints, the standalone head can be inspected in Python:

```python
import torch
from sf_xxl.dmlab.world_model import DGEventWorldModel

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
saved = checkpoint["dg_world_model"]
cfg = saved["config"]
model = DGEventWorldModel(**{k: v for k, v in cfg.items() if k != "horizon"})
model.load_state_dict(saved["model"])
model.eval()
# real_ca3: [B, F*(R+L-1)], AFTER processing the current observation,
# using the matching checkpoint's DG detector and real observation history.
predictions = model.predict_actions(real_ca3, cfg["horizon"])
# Each output has shape [B, number_of_actions, ...]. No real state is modified.
```

No reward/termination model, imagined Bellman backup, or policy-distillation loss
is implemented. Episode boundaries censor rather than supply terminal-event
labels. Predicting CA3 alone also does not reconstruct bypass observations. Add
these deliberately if the next experiment requires planning during learning.

## Verification and reusable lessons

From the repository root, in the existing SF environment:

```bash
python -m pytest -q tests/algo/test_dg_world_model.py
```

Tests exercise the actual DefaultLearner PPO loop and existing BypassSS core on
CPU with a small synthetic DG encoder: enabled/disabled PPO weight equality,
separate predictor updates, packed replay with an episode reset, checkpoint
save/startup resume and optimizer continuation, action conditioning, multi-DG
events, censored labels, and invalid action sentinels. No DMLab performance or
transfer improvement is claimed by these tests.

Use the runtime hook rather than copying a learner: this preserves optimizer,
masking and checkpoint behavior. Inspect the actual launcher imports before
choosing integration files. During testing, the synthetic policy-version field
needed SF's floating-point storage convention; compare complete learner updates
rather than testing the head alone to catch these interface mismatches.
