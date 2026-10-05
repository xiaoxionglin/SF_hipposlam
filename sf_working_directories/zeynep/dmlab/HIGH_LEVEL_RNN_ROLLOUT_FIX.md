# High-level RNN rollout and trial-history fix

This note explains the changes to `exp_ymaze_HLRNN.py` for Zeynep. The high-level
model still acts from its recurrent hidden state $h$. The added trial records are
used **only by the learner** to train recurrent updates across 64-frame rollout
boundaries; the Q head does not read them during acting.

## What was wrong

1. Sample Factory carries `rnn_states` between rollouts, but backpropagation
   stops at `recurrence=64`. A reward update near the end of one rollout could
   have no later high-level loss in the same training chunk, so that update
   received no useful gradient from the next trial.
2. The actor's sampled mode was not saved as a trajectory field. During learner
   replay, the high-level module sampled again. The reconstructed decoder input
   could differ from the mode used for the recorded low-level action, and a
   later reward in the chunk could be assigned to the replayed mode.
3. Sampling was tied to `torch.is_grad_enabled()`. Sample Factory's inference
   workers collect training rollouts without gradients, so this made the actor
   choose greedily during data collection.
4. The experiment's name said `Q_grid`, but `--hl_is_policy=True` selected its
   policy-gradient loss. This run now explicitly uses the intended Q regression.

A negative `hl_loss` in that earlier policy run is possible without numerical
error: the policy objective subtracts $0.05$ times the mode entropy. With four
uniform modes and zero advantage, it is $-0.05\log 4 \approx -0.0693$. A finite
Q-regression loss is a masked mean of squared errors and cannot be negative. If
this experiment reports a negative high-level loss after switching to
`--hl_is_policy=False`, check the effective launch arguments and run directory.

## Data and timing

At a center outcome event, the old mode $z_{k-1}$ earned the latched reward
$r_{k-1}$. The actor records the pair $(z_{k-1},r_{k-1})$, updates $h$, samples
$z_k$, and passes $z_k$ to the low-level decoder. The first center event has no
previously chosen mode, so it does not enter the completed-trial history or the
high-level loss.

The actor's recurrent state now contains, in order:

```text
base_core_state | high_level_h | last N completed trial records | current_z
```

Each record contains the hidden state **before** that reward update, the mode,
the reward, and a validity flag. Records shift only at completed-trial events.
They persist across 64-frame rollouts and are cleared when Sample Factory
reports a real environment `done`. The actor's Q head still receives only
`high_level_h`; the record fields cannot directly tell it which arm to choose.

For each outcome in learner replay, the learner starts from the oldest saved
pre-event hidden state, unrolls the RNN through up to `hl_history_len` earlier
trial outcomes, and predicts the value of the **actor's** mode for the current
reward. This gives the RNN-cell parameters a gradient through earlier reward
updates even when those events occurred before the current 64-frame rollout.
The starting hidden state is detached, so the trial history is a bounded
truncated-backpropagation window. No second PPO implementation is involved.

## Code changes

| File | Change |
| --- | --- |
| `sample_factory/algo/utils/shared_buffers.py` | Allocate `hl_z` for the high-level core only. |
| `custom_actor_critic.py` | Save the post-decision mode from `new_rnn_states` as `hl_z` alongside the actor's actions; use a reproducible greedy mode for value-only bootstrap calls. |
| `sample_factory/algo/learning/learner.py` | Pass recorded `hl_z` and training-only trial records into the packed recurrent learner input. |
| `custom_core.py` | Carry the bounded trial history, unroll it at outcome events, use recorded modes during replay, and reject an incorrect configured state size. |
| `custom_highlevelRNN.py` | Separate reward-state updates from mode sampling; sample or choose greedily using an explicit setting rather than autograd state. |
| `custom_params.py` | Add `hl_history_len` and `hl_deterministic`. |
| `exp_ymaze_HLRNN.py` | Set eight historical trials, update `rnn_size` from 1166 to 1342, and select Q regression. |

The learner also passes its `valids` flag through the packed sequence. The
high-level event mask now excludes invalid samples, matching the ordinary PPO
losses. Computing the high-level loss inside the core is functional here: the
learner reads `last_hl_loss` after the core forward pass, adds it to the total
loss, and calls `backward()` on that total. The mutable attribute is a coupling
to the current one-core-forward-per-minibatch flow; if that flow changes, return
the auxiliary loss explicitly from the forward pass instead.

For this configuration, eight records use $8(16+4+2)=176$ state elements.
`hl_deterministic=False` is the training default. Set it to `True` only when
greedy high-level evaluation is intended; actor inference mode alone no longer
changes exploration. The value-only bootstrap path temporarily chooses greedily
to avoid adding random sampling noise to the next-value estimate. Other runs
using the high-level core must update their
`rnn_size` to account for `hl_history_len` or the core will raise a size error.

## Validation and limits

`tests/test_high_level_rollout.py` checks sampling under `no_grad`, history
continuity across separate forward calls, a nonzero recurrent gradient through
an earlier trial, and decoder replay under a recorded mode. These are unit
checks. Run them with `python -m pytest -q tests/test_high_level_rollout.py`.
An end-to-end DeepMind Lab run is still needed to inspect event timing,
reward alignment, mode use, and learning curves. The configured level
`ymaze_vol5_INSTR_HL` is not checked into this branch, so its Lua termination
behavior cannot be verified here. The Python wrapper only reports `done` when
DeepMind Lab stops running.

The new state layout is incompatible with old high-level checkpoints. Start a
new experiment rather than resuming a 1166-element state. Saved pre-event
hidden states were produced by actor parameters at collection time; if policy
lag is large, reconstructed histories can differ from those states. Keep the
window short, inspect policy lag, and compare against a `hl_history_len=1`
control. The Q loss still uses only the chosen mode and immediate completed
trial reward. It does not implement multi-trial return or off-policy
correction.
