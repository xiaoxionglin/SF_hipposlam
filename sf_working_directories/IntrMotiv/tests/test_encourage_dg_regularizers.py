import math

import torch

from sf_working_directories.IntrMotiv.dmlab.custom_learner import dg_usage_metrics
from sf_working_directories.IntrMotiv.dmlab.experiments.encourage_dg_regularizers import (
    LOSS_ARMS,
    RUN_DESCRIPTION,
)


def test_encourage_regularizer_factorial_is_exactly_replicated():
    experiments = RUN_DESCRIPTION.experiments
    assert len(experiments) == 60
    names = [experiment.base_name for experiment in experiments]
    assert len(names) == len(set(names))
    assert {arm.tag for arm in LOSS_ARMS} == {"CTRL", "G001", "G003", "R100", "G001_R100"}
    for experiment in experiments:
        assert "--encoder_reward_method=encourage" in experiment.cmd
        assert "--encoder_batch_loss=True" in experiment.cmd
        assert "--train_for_env_steps=100000000" in experiment.cmd
        assert "--iterative_update=False" in experiment.cmd


def test_dg_usage_metrics_report_dead_units_and_normalized_entropy():
    dg_active = torch.tensor([[True, False, False], [True, True, False]])
    minimum, mean, maximum, entropy = dg_usage_metrics(dg_active)
    expected_entropy = -((2 / 3) * math.log(2 / 3) + (1 / 3) * math.log(1 / 3)) / math.log(3)
    assert minimum.item() == 0.0
    assert mean.item() == 0.5
    assert maximum.item() == 1.0
    assert math.isclose(entropy.item(), expected_entropy, rel_tol=1e-6)
