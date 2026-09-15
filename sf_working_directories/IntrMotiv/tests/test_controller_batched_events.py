from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from sf_working_directories.IntrMotiv.dmlab.controller_transition import (
    ReplayRejected,
    TransitionInput,
    transition_values,
    transition_values_batch,
)
from sf_working_directories.IntrMotiv.tests.test_controller_transition import RewardModel, physical_rows


@pytest.mark.parametrize("write", [False, True])
def test_mixed_main_her_terminal_and_rejections_match_scalar_events_and_gradients(write):
    model = RewardModel(write).eval()
    other = deepcopy(model)
    rows, _ = physical_rows(model)
    first = TransitionInput(tuple(rows[:2]), 0)
    terminal = TransitionInput((rows[0], replace(rows[1], truncated=True), rows[2]), 1)
    examples = [
        first,
        replace(first, virtual_goal=1, remaining=3),
        terminal,
        replace(first, virtual_goal=2, remaining=1),
        replace(first, virtual_goal=0, remaining=3),
        replace(first, rows=(replace(rows[0], generation=5), rows[1])),
    ]
    separate = []
    for example in examples:
        try:
            separate.append(transition_values(other, example))
        except ReplayRejected as exc:
            separate.append(str(exc))
    together = transition_values_batch(model, examples)
    for a, b in zip(separate, together):
        if isinstance(a, str):
            assert a == b
            continue
        assert a["done"] == b["done"]
        for key in ("q", "reward", "signature", "canonical", "next_canonical"):
            torch.testing.assert_close(a[key], b[key])
    sum(r["q"].square().mean() for r in separate if not isinstance(r, str)).backward()
    sum(r["q"].square().mean() for r in together if not isinstance(r, str)).backward()
    for a, b in zip(model.parameters(), other.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)
