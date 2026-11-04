## Installation

Main dmlab package needs to be built from sources (see dmlab github repo)

Besides:

```shell
sudo apt install libosmesa6-dev
pip install dm_env
```

## Openfield episode semantics

Use `openfield_map2_fixed_loc3_fixedlength_noreward` for intrinsic-motivation
experiments that require a fixed episode horizon. Goal contact does not end this
level; its existing 120-second timeout ends every episode after 7,200 engine
frames. With `--env_frameskip=8`, this is 900 agent actions.

The older `openfield_map2_fixed_loc3_noreward` level ends when the zero-reward
goal is touched, so its episode lengths vary with policy behavior. It remains
registered for compatibility with existing runs.

After adding or changing custom Lua levels, install the repository patch into
the active environment with `deepmindlab_patch/patch_deepmindlab.sh`. The
`SFgit` environment on NEMO2 already contains the fixed-length level.

## Reward summaries

`reward/reward` is Sample Factory's episodic environment return. It is expected
to remain zero in a no-reward level. IntrMotiv also records the reward streams
that actually drive learning:

- `reward/learning_*`: reward used to calculate policy advantages.
- `reward/intrinsic_*`: target-gated intrinsic reward.
- `reward/environment_*`: raw environment reward seen by the learner.

The detailed learner metrics remain available under
`train/reward_for_advantage_*`, `train/intrinsic_reward_*`, and
`train/env_reward_*`.

## Matched task-cue transfer initialization

For a controlled comparison of flat task conditioning, use
`--fixed_task_conditioning=True --fixed_task_goal_init=matched_cues` and vary
only `--fixed_task_goal_mixture=True|False`. Both modes start from identical,
seeded near-uniform effective task vectors and the same fresh FiLM parameters.
The mixture stores log probabilities and applies softmax; the unrestricted
mode stores those initial probabilities directly and may subsequently leave
the simplex. Five-cue tasks use one independently initialized vector per cue.
This opt-in mode leaves all historical initialization settings unchanged.

To isolate worker reuse, cross this switch with `--transfer_scope=dg|policy`
using the same source checkpoint, `--transfer_freeze_dg=True`, and
`--transfer_freeze_worker=False`. Policy transfer replaces the decoder and
action head with source weights; both scopes keep a fresh reward critic and
optimizer. Use external PPO reward, disable graph transfer and HRL, and retain
`hrl_goal_conditioning=target_id_film` with immediate target timing. The task
vector's gradient remains live with `--ppo_dg_gradient=stop`; DG weights and
BatchNorm statistics stay frozen.

Matched initialization does not match optimization geometry: softmax changes
the effective task-vector gradient and constrains its future values. Report
this as a task-conditioning parameterization comparison, not as proof that
the simplex constraint alone caused an outcome difference.

## Matched layer2 ResNet + LSTM baseline

Use `--layer2_lstm_baseline=sparse` or `--layer2_lstm_baseline=dense` on an
externally rewarded level. Both modes reuse IntrMotiv's frozen ImageNet ResNet-18
through layer2, number-instruction encoding, capped inverse-depth path, DG-width
linear projection with BatchNorm, decoder, action interface, and ordinary PPO.
The first `Hippo_n_feature` projected values enter a standard LSTM. The depth
and cue bypass goes directly to the decoder on each step, matching the BypassSS
input routing. The switch selects `core_name=LstmDGBypass`, `rnn_type=lstm`,
`encoder_conv_architecture=layer2_resnet18`, and joint PPO gradients. A legacy
`--rnn_size=0` becomes a 256-unit LSTM; a positive explicit size is retained.

`sparse` uses the existing thresholded ReLU DG output and `DG_BN_intercept`
(default 2). `dense` returns the signed normalized linear projection before
thresholding. Projection width, weights, and BatchNorm implementation are the
same in both arms. The sparse/dense flag defaults to `off`, preserving older
runs and checkpoints.

The switch disables the default distance learner and DG-only online spatial
telemetry when those options were not explicitly requested. It rejects explicit
internal-reward, distance-learning, DG telemetry, CA3/HRL, and incompatible DG
settings. For comparable runs, use the same externally rewarded level, RGBD and
cue settings, PPO hyperparameters, seeds, frame budget, checkpoints, and
evaluation protocol. Verify that the selected task supplies external reward;
ordinary PPO has no learning signal from an all-zero reward stream. The switch
does not reproduce IntrMotiv's DG-derived intrinsic reward or CA3 graph
mechanisms.
