# Standalone SimGaY manager

This optional entry point runs the completed-trial high-level RNN with PyTorch
and NumPy. It does not import Sample Factory, DeepMind Lab or TorchVision.
Existing navigation entry points and defaults are unchanged.

```bash
python -m sf_examples.simgay.train --help
python -m sf_examples.simgay.train --output train_dir/simgay_seed7
```

The default task uses five 15-trial blocks, 80/20 rewards, a 0.66 switch
probability at nominal boundaries, and a perfect actuator. Set
`--actuator-success-rate 0.8` explicitly for the noisy actuator condition,
including when evaluating its checkpoint. The actuator uses a separate random
stream. The manager observes commanded modes and rewards; executed arms and
hidden task state enter diagnostic metrics only.

The default manager uses Q loss, two modes, 16 hidden units, eight completed
trials of replay and temperature 1. `--loss policy` selects the existing policy
loss. `--init-checkpoint PATH` initializes compatible weights;
`--evaluate-checkpoint PATH` runs frozen evaluation. Both require a new
`--output` directory. Evaluation separates sampled, greedy and reset-memory
policies, and reports random, win-stay/lose-shift and oracle baselines.

Source: `SF_hipposlam_hl_bandit`, branch `codex/standalone-high-level-bandit`,
commit `bc2bcb22483f82a8dbf133a0a9a0102f0d292c1f`. The manager, trial history,
loss definitions, environment, runner and checkpoint contracts were retained;
imports and the repository-root lookup were adapted to this package.
Older unused manager variants and the full navigation/oracle-transfer core
were not imported. Checkpoint tests validate the manager/wrapper tensor
protocol; they do not establish full DeepMind Lab transfer or that a learned
navigation controller realizes the bandit's fixed mode-to-arm mapping.

Focused checks:

```bash
python -m pytest -q tests/test_simgay_bandit_environment.py tests/test_simgay_bandit_training.py tests/test_simgay_bandit_checkpoints.py tests/test_simgay_trial_history.py
```
