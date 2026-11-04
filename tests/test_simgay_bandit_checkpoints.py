"""Standalone manager checkpoints and the supported wrapper tensor protocol.

The lightweight wrapper here checks checkpoint compatibility, not DMLab transfer.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from sf_examples.simgay.checkpoints import (
    build_model_from_checkpoint,
    load_checkpoint_into,
    read_checkpoint,
    save_checkpoint,
)
from sf_examples.simgay.manager import HighLevelContextRNN_Learner


def make_full_model(core_name="core", **overrides):
    cfg = SimpleNamespace(
        Hippo_n_feature=2,
        Hippo_R=1,
        Hippo_L=2,
        hl_K=2,
        hl_d_H=3,
        hl_history_len=2,
        hl_tau=0.7,
        hl_is_policy=False,
        hl_representations=False,
        hl_d_rep=4,
        oracle_context=False,
    )
    vars(cfg).update(overrides)
    full = nn.Module()
    full.cfg = cfg
    full.encoder = nn.Linear(3, 4)
    core = nn.Module()
    core.history_len = cfg.hl_history_len
    core.hl_learner = HighLevelContextRNN_Learner(
        K=cfg.hl_K,
        d_H=cfg.hl_d_H,
        tau=cfg.hl_tau,
        is_policy=cfg.hl_is_policy,
        representations=cfg.hl_representations,
        d_rep=cfg.hl_d_rep,
    )
    core.unrelated = nn.Linear(2, 3)
    setattr(full, core_name, core)
    full.decoder = nn.Linear(4, 2)
    return full


def manager_like(full):
    source = full.core.hl_learner
    return HighLevelContextRNN_Learner(
        K=source.K,
        d_H=source.d_H,
        tau=source.tau,
        is_policy=source.is_policy,
        representations=source.representations,
        d_rep=source.d_rep,
    )


@pytest.mark.parametrize("representations", [False, True])
def test_standalone_round_trip_preserves_exact_manager_and_metadata(tmp_path, representations):
    source = HighLevelContextRNN_Learner(
        K=2,
        d_H=3,
        tau=0.7,
        is_policy=True,
        representations=representations,
        d_rep=4,
    )
    path = save_checkpoint(tmp_path / "manager.pth", source, 2, [1, 0], extra={"trials": 100})
    restored, metadata = build_model_from_checkpoint(path, expected_mode_to_arm=[1, 0])
    assert metadata == dict(
        K=2,
        d_H=3,
        tau=0.7,
        is_policy=True,
        representations=representations,
        d_rep=4,
        history_len=2,
        mode_to_arm=[1, 0],
    )
    for key, tensor in source.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])
    payload = torch.load(path, weights_only=False)
    assert payload["weights_only"] is True
    assert "optimizer" not in payload
    assert payload["extra"] == {"trials": 100}


@pytest.mark.parametrize("core_name", ["core", "actor_core"])
def test_manager_load_into_full_wrapper_preserves_all_other_tensors(tmp_path, core_name):
    full = make_full_model(core_name=core_name)
    core = getattr(full, core_name)
    source = HighLevelContextRNN_Learner(K=2, d_H=3, tau=0.7, d_rep=4)
    path = save_checkpoint(tmp_path / "manager.pth", source, 2, [0, 1])
    before = {key: tensor.clone() for key, tensor in full.state_dict().items()}
    load_checkpoint_into(full, path)
    for key, tensor in full.state_dict().items():
        if key.startswith(f"{core_name}.hl_learner."):
            assert torch.equal(tensor, source.state_dict()[key.split("hl_learner.", 1)[1]])
        else:
            assert torch.equal(tensor, before[key])
    # The same shared model produces the same recurrent states and decisions.
    h, z, reward = torch.randn(5, 3), torch.eye(2)[torch.arange(5) % 2], torch.randn(5)
    mask = torch.ones(5, dtype=torch.bool)
    expected = source.update_state(mask, reward, h, z)
    actual = core.hl_learner.update_state(mask, reward, h, z)
    assert torch.equal(expected, actual)
    assert torch.equal(source.head(expected), core.hl_learner.head(actual))


@pytest.mark.parametrize("core_name", ["core", "actor_core"])
def test_full_sf_checkpoint_import_reads_config_and_preserves_unknown_mapping(tmp_path, core_name):
    source = make_full_model(core_name=core_name, hl_is_policy=True)
    policy_dir = tmp_path / "checkpoint_p0"
    policy_dir.mkdir()
    path = policy_dir / "checkpoint_100.pth"
    torch.save({"model": source.state_dict(), "train_step": 100, "optimizer": {}}, path)
    (tmp_path / "config.json").write_text(json.dumps(vars(source.cfg)), encoding="utf-8")
    restored, metadata = build_model_from_checkpoint(path)
    assert metadata["is_policy"] is True
    assert metadata["mode_to_arm"] is None
    for key, tensor in getattr(source, core_name).hl_learner.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[key])


def test_full_sf_import_requires_semantic_config(tmp_path):
    source = make_full_model()
    path = tmp_path / "full.pth"
    torch.save({"model": source.state_dict()}, path)
    with pytest.raises(ValueError, match="requires its config"):
        read_checkpoint(path)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"is_policy": True}, "is_policy"),
        ({"tau": 0.2}, "tau"),
        ({"representations": True}, "representations"),
        ({"d_rep": 8}, "d_rep"),
        ({"d_H": 4}, "d_H"),
        ({"K": 3}, "K"),
    ],
)
def test_semantic_and_architecture_mismatches_fail_before_mutation(tmp_path, change, message):
    target = make_full_model()
    kwargs = dict(K=2, d_H=3, tau=0.7, d_rep=4)
    kwargs.update(change)
    source = HighLevelContextRNN_Learner(**kwargs)
    path = save_checkpoint(tmp_path / "source.pth", source, 2, [0] * source.K)
    before = {key: tensor.clone() for key, tensor in target.state_dict().items()}
    with pytest.raises(ValueError, match=message):
        load_checkpoint_into(target, path)
    assert all(torch.equal(value, target.state_dict()[key]) for key, value in before.items())


def test_history_and_known_mode_mapping_mismatch_are_rejected(tmp_path):
    target = make_full_model()
    source = manager_like(target)
    path = save_checkpoint(tmp_path / "source.pth", source, 3, [1, 0])
    with pytest.raises(ValueError, match="history_len"):
        load_checkpoint_into(target, path)
    with pytest.raises(ValueError, match="mode_to_arm"):
        load_checkpoint_into(target.core.hl_learner, path, expected_mode_to_arm=[0, 1])


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape"])
def test_tensor_validation_is_atomic(tmp_path, corruption):
    target = make_full_model()
    source = manager_like(target)
    path = save_checkpoint(tmp_path / "source.pth", source, 2, [0, 1])
    payload = torch.load(path, weights_only=False)
    if corruption == "missing":
        del payload["model"]["head.bias"]
    elif corruption == "unexpected":
        payload["model"]["unexpected"] = torch.zeros(2)
    else:
        payload["model"]["head.bias"] = torch.zeros(3)
    torch.save(payload, path)
    before = {key: tensor.clone() for key, tensor in target.state_dict().items()}
    with pytest.raises(ValueError, match="Incompatible high-level tensor"):
        load_checkpoint_into(target, path)
    assert all(torch.equal(value, target.state_dict()[key]) for key, value in before.items())
