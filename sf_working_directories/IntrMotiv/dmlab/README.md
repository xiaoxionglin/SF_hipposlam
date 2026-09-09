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
