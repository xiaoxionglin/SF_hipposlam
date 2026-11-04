"""Shared DG versus the retained IntrMotiv implementation, without vision imports."""

import ast
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sf_shared.dg import DGProjection_batchnorm_relu as SharedDG


def legacy_class():
    # Compile the actual retained class only: importing its containing module
    # would load unrelated torchvision/DMLab dependencies into these tests.
    path = Path(__file__).resolve().parents[1] / "sf_working_directories/IntrMotiv/dmlab/custom_encoder.py"
    node = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == SharedDG.__name__
    )
    namespace = dict(torch=torch, nn=nn, F=F, contextmanager=contextmanager)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[node.name]


LegacyDG = legacy_class()
MODES = ("legacy_batch", "running_consistent", "running_poststep_atomic", "input_centered_atomic")


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def assert_state_equal(left, right):
    assert left.state_dict().keys() == right.state_dict().keys()
    for key, value in left.state_dict().items():
        torch.testing.assert_close(value, right.state_dict()[key], rtol=0, atol=0)


def pair(mode, activation="threshold_relu"):
    torch.manual_seed(91)
    old = LegacyDG(6, 4, intercept=0.2, batchnorm_semantics=mode, output_activation=activation)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(91)
    new = SharedDG(6, 4, intercept=0.2, batchnorm_semantics=mode, output_activation=activation)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert_state_equal(old, new)
    return old, new


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("activation", ("threshold_relu", "identity"))
@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("update", (False, True))
def test_forward_gradients_and_poststep_statistics_match(mode, activation, training, update):
    old, new = pair(mode, activation)
    inputs = torch.randn(7, 6)
    input_grads = []
    outputs = []
    for model in (old, new):
        model.train(training)
        x = inputs.clone().requires_grad_()
        with model.running_stats_update(update):
            output = model(x)
        output.square().sum().backward()
        outputs.append(output.detach())
        input_grads.append(x.grad)
        with torch.no_grad():
            model.linear.weight.add_(model.linear.weight.grad, alpha=-0.01)
        if training and update and mode.endswith("atomic"):
            assert model.post_step_update_running_stats(torch.tensor([1, 1, 0, 1, 1, 0, 1]))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(input_grads[0], input_grads[1], rtol=0, atol=0)
    torch.testing.assert_close(old.linear.weight.grad, new.linear.weight.grad, rtol=0, atol=0)
    assert_state_equal(old, new)
    assert old.last_running_stats_updated == new.last_running_stats_updated
    assert old.last_poststep_calibration_updated == new.last_poststep_calibration_updated
    before = {name: value.clone() for name, value in new.state_dict().items()}
    old.eval()
    new.eval()
    torch.testing.assert_close(old(inputs[:1]), new(inputs[:1]), rtol=0, atol=0)
    for name, value in new.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("calibrate", (False, True))
def test_frozen_statistics_and_one_time_calibration_match(mode, calibrate):
    old, new = pair(mode)
    inputs = torch.randn(7, 6)
    for model in (old, new):
        model.freeze_running_stats = True
        model.calibrate_frozen_on_first_batch = calibrate
        with model.running_stats_update(True):
            model(inputs)
        before = {name: value.clone() for name, value in model.state_dict().items()}
        with model.running_stats_update(True):
            model(inputs + 5)
        assert int(model.batchnorm1d.num_batches_tracked) == int(calibrate)
        assert not model.last_running_stats_updated
        for name, value in model.state_dict().items():
            assert torch.equal(value, before[name])
    assert_state_equal(old, new)
    torch.testing.assert_close(old(inputs), new(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("partial_recruitment", (False, True))
def test_old_checkpoint_loading_and_bidirectional_round_trip(mode, partial_recruitment):
    old, new = pair(mode)
    old = nn.Sequential(old)
    new = nn.Sequential(new)
    checkpoint = {
        key: value.clone() for key, value in old.state_dict().items() if key.startswith(("0.linear.", "0.batchnorm1d."))
    }
    if partial_recruitment:
        checkpoint["0.recruitment_committed"] = torch.tensor([True, False, True, False])
    old.load_state_dict(dict(checkpoint), strict=True)
    new.load_state_dict(dict(checkpoint), strict=True)
    assert_state_equal(old, new)
    expected = checkpoint.get("0.recruitment_committed", torch.zeros(4)).long()
    assert torch.equal(new[0].recruitment_row_counts, expected)
    with torch.no_grad():
        new[0].linear.weight.add_(1)
    new[0].mark_replacement_committed()
    old.load_state_dict(new.state_dict(), strict=True)
    new.load_state_dict(old.state_dict(), strict=True)
    assert_state_equal(old, new)


@pytest.mark.parametrize("cls", (LegacyDG, SharedDG))
def test_context_and_calibration_failure_contract(cls):
    model = cls(6, 4, batchnorm_semantics="running_poststep_atomic")
    with pytest.raises(RuntimeError, match="cached projection"):
        model.post_step_update_running_stats()
    with pytest.raises(ValueError, match="sentinel"):
        with model.running_stats_update(True):
            with model.running_stats_update(False):
                assert not model._running_stats_update_enabled
            assert model._running_stats_update_enabled
            raise ValueError("sentinel")
    assert not model._running_stats_update_enabled
    with model.running_stats_update(True):
        model(torch.randn(3, 6))
    with pytest.raises(ValueError, match="align"):
        model.post_step_update_running_stats(torch.ones(2))
    with model.running_stats_update(True):
        model(torch.randn(3, 6))
    with pytest.raises(RuntimeError, match="at least two"):
        model.post_step_update_running_stats(torch.tensor([1, 0, 0]))
    assert int(model.weight_generation) == 0


def test_shared_import_does_not_load_sample_factory_or_vision():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import sf_shared.dg; "
            "assert not any(k.startswith(('sample_factory', 'deepmind_lab', 'torchvision', 'gymnasium')) "
            "for k in sys.modules)",
        ],
        check=True,
    )
