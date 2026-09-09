from types import SimpleNamespace

from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
from sf_working_directories.IntrMotiv.dmlab.iterative_update import (
    DECODER,
    ENCODER,
    SIMULTANEOUS,
    IterativeUpdateSchedule,
)


def test_iterative_schedule_has_warmup_and_four_to_one_cycle():
    schedule = IterativeUpdateSchedule(enabled=True, initial_encoder_steps=2, decoder_steps=4, encoder_steps=1)

    assert [schedule.phase(step) for step in range(9)] == [
        ENCODER,
        ENCODER,
        DECODER,
        DECODER,
        DECODER,
        DECODER,
        ENCODER,
        DECODER,
        DECODER,
    ]
    assert IterativeUpdateSchedule(enabled=False).phase(0) == SIMULTANEOUS


def test_learner_preserves_the_configured_phase_schedule_without_gradient_masks():
    learner = object.__new__(DistanceLearnerReward)
    learner.cfg = SimpleNamespace(
        iterative_update=True,
        iterative_initial_encoder_steps=2,
        iterative_decoder_steps=4,
        iterative_encoder_steps=1,
        iterative_start_phase=DECODER,
    )
    assert [(setattr(learner, "train_step", step), learner._iterative_phase())[1] for step in range(7)] == [
        ENCODER,
        ENCODER,
        DECODER,
        DECODER,
        DECODER,
        DECODER,
        ENCODER,
    ]
    assert not hasattr(learner, "_apply_iterative_gradient_mask")
