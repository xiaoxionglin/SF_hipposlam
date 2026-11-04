"""Real tiny checkpoint/event fixtures; no DMLab or external source required."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from hpc_runs.intrmotiv_study.spec import StudySpec
from hpc_runs.intrmotiv_study.transfer_audit import TASK, audit


def _save(path, model, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model, "env_steps": frames, "train_step": 0 if frames == 0 else 1}, path)


def _events(run_dir, *, reward=0.25, mismatch=False, omit=None):
    with SummaryWriter(str(run_dir / ".summary" / "0")) as writer:
        for step in (16, 32):
            for tag, value in {
                "train/env_steps": step,
                "intrmotiv/reward/advantage_mean": reward + float(mismatch),
                "intrmotiv/reward/environment_mean": reward,
                "train/policy_loss": 0.1,
            }.items():
                if tag != omit:
                    writer.add_scalar(tag, value, step)


def fixture(tmp_path, scope="policy", *, events=True):
    source_path = tmp_path / "source" / "checkpoint_p0" / "checkpoint_000000001_100.pth"
    source = {
        "encoder.DG_projection.linear.weight": torch.arange(15.0).reshape(3, 5),
        "encoder.DG_projection.batchnorm1d.running_mean": torch.arange(3.0),
        "encoder.DG_projection.batchnorm1d.running_var": torch.ones(3),
        "encoder.DG_projection.batchnorm1d.num_batches_tracked": torch.tensor(7),
        "encoder.DG_projection.weight_generation": torch.tensor(0),
        "decoder.state.weight": torch.arange(24.0).reshape(4, 6),
        "action_parameterization.weight": torch.arange(8.0).reshape(2, 4),
        "critic_linear.weight": torch.ones(1, 4),
    }
    _save(source_path, source, 100)
    source_config = {
        "env_frameskip": 4, "core_name": "ExampleCA3", "Hippo_n_feature": 3,
        "Hippo_R": 2, "Hippo_L": 4, "encoder_conv_architecture": "example",
        "depth_sensor": True, "normalize_input": False, "hrl_goal_conditioning": "target_id_film",
        "dmlab_navigation_action_set": True,
    }
    (source_path.parents[1] / "config.json").write_text(json.dumps(source_config))
    config = {**source_config, "env": "arbitrary_reward_task", "depth_sensor_inverse": False,
              "transfer_scope": scope, "transfer_model_path": str(source_path), "transfer_freeze_dg": True,
              "transfer_freeze_worker": False, "advantage_reward_source": "external", "num_policies": 1,
              "with_pbt": False, "extra_encoder_losses": False, "fixed_task_conditioning": True,
              "fixed_task_goal_mixture": False, "reward_instruction_count": 2,
              "hrl_controllable_graph": False, "reward_scale": 0.1, "train_for_env_steps": 64, "seed": 42}
    raw = {
        "schema": "intrmotiv/study/v1", "workflow_version": "1.14.1", "study_id": "transfer_fixture",
        "description": "Synthetic qualification audit fixtures",
        "training": {"batch_name": "batch", "workspace_root": str(tmp_path), "output_root": str(tmp_path / "batch"),
                     "run_name_template": "{base}_S{seed}", "condition_name_template": "{base}",
                     "common_args": [f"--{key}={value}" for key, value in config.items() if key != "seed"]},
        "bases": [{"name": "TRANSFER", "metadata": {"source_checkpoint": str(source_path),
                   "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()}}], "seeds": [42],
    }
    study = StudySpec.from_mapping(raw)
    run_dir = tmp_path / "batch" / "TRANSFER_S42_" / "00_TRANSFER_S42"
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(config))
    before = {key: value.clone() for key, value in source.items()}
    before["critic_linear.weight"] += 2
    if scope == "dg":
        before["decoder.state.weight"] += 1
        before["action_parameterization.weight"] += 1
    before[TASK] = torch.ones(2, 3) / 3
    after = {key: value.clone() for key, value in before.items()}
    after["decoder.state.weight"] += 0.01
    after["critic_linear.weight"] += 0.01
    after[TASK][:, 0] += torch.tensor([0.01, 0.02])
    initial = run_dir / "checkpoint_p0" / "initial_000000000_0.pth"
    terminal = run_dir / "checkpoint_p0" / "milestones" / "checkpoint_000000001_64.pth"
    _save(initial, before, 0)
    _save(terminal, after, 64)
    if events:
        _events(run_dir)
    return SimpleNamespace(study=study, root=tmp_path / "batch", config=config, run_dir=run_dir,
                           before=before, after=after, initial=initial, terminal=terminal, source=source,
                           source_path=source_path)


def result(f):
    return audit(f.study, f.root)


def failure(f, message):
    report = result(f)
    assert not report["passed"]
    assert message in " ".join(report["runs"][0]["errors"])


@pytest.mark.parametrize("scope", ["policy", "dg"])
def test_real_checkpoints_and_tensorboard_pass_for_arbitrary_dimensions(tmp_path, scope):
    f = fixture(tmp_path, scope)
    report = result(f)
    assert report["passed"], report
    row = report["runs"][0]
    assert row["checkpoint_frames"] == 64 and row["positive_reward_batches"] == 2
    assert len(row["task_movement_l1_per_cue"]) == 2
    assert row["max_logged_frames"] == 32  # Last log can precede final checkpoint.
    assert report["study_sha256"] == f.study.fingerprint


def test_initial_checkpoint_is_required_not_substituted_by_terminal(tmp_path):
    f = fixture(tmp_path)
    f.initial.unlink()
    failure(f, "expected one initial checkpoint")


def test_legacy_frame_zero_and_milestone_terminal_are_supported(tmp_path):
    f = fixture(tmp_path)
    f.initial.rename(f.initial.with_name("checkpoint_000000000_0.pth"))
    assert result(f)["passed"]


@pytest.mark.parametrize("key", ["encoder.DG_projection.batchnorm1d.running_mean",
                                  "encoder.DG_projection.batchnorm1d.num_batches_tracked",
                                  "encoder.DG_projection.weight_generation"])
def test_any_frozen_projection_buffer_drift_fails(tmp_path, key):
    f = fixture(tmp_path)
    f.after[key] += 1
    _save(f.terminal, f.after, 64)
    failure(f, "frozen DG parameters/BN/buffers changed")


@pytest.mark.parametrize("change,message", [
    ("missing_bn", "missing compared tensor"), ("source_dg", "source DG"),
    ("worker_load", "worker transfer mismatch"), ("worker_stuck", "worker did not change"),
    ("critic_copy", "critic was not freshly initialized"), ("cue_stuck", "every cue"),
    ("nonfinite", "nonfinite task parameters"), ("cue_shape", "task parameter shape"),
])
def test_tensor_contract_failures(tmp_path, change, message):
    f = fixture(tmp_path)
    if change == "missing_bn":
        del f.before["encoder.DG_projection.batchnorm1d.running_mean"]
    elif change == "source_dg":
        f.before["encoder.DG_projection.linear.weight"] += 1
    elif change == "worker_load":
        f.before["decoder.state.weight"] += 1
    elif change == "worker_stuck":
        f.after["decoder.state.weight"] = f.before["decoder.state.weight"].clone()
    elif change == "critic_copy":
        f.before["critic_linear.weight"] = f.source["critic_linear.weight"].clone()
    elif change == "cue_stuck":
        f.after[TASK][1] = f.before[TASK][1]
    elif change == "nonfinite":
        f.after[TASK][0, 0] = float("nan")
    else:
        f.after[TASK] = f.after[TASK][0]
    _save(f.initial, f.before, 0)
    _save(f.terminal, f.after, 64)
    failure(f, message)


def test_dg_only_transfer_rejects_copied_worker(tmp_path):
    f = fixture(tmp_path, "dg")
    f.before["decoder.state.weight"] = f.source["decoder.state.weight"].clone()
    _save(f.initial, f.before, 0)
    failure(f, "fresh worker has no changed source weights")


@pytest.mark.parametrize("variant,message", [("missing", "missing TensorBoard"),
    ("zero", "positive external reward batches"), ("mismatch", "reward mismatch"),
    ("tag", "missing TensorBoard tags"), ("nan", "nonfinite learning scalar")])
def test_reward_data_fail_closed(tmp_path, variant, message):
    f = fixture(tmp_path, events=False)
    if variant != "missing":
        _events(f.run_dir, reward=0 if variant == "zero" else float("nan") if variant == "nan" else 0.25,
                mismatch=variant == "mismatch", omit="intrmotiv/reward/environment_mean" if variant == "tag" else None)
    failure(f, message)


def test_hash_and_source_depth_interface_are_checked(tmp_path):
    f = fixture(tmp_path)
    with f.source_path.open("ab") as handle:
        handle.write(b"changed")
    failure(f, "SHA-256 mismatch")
    # Restore the hash, then alter only the source interface (not tensor shapes).
    _save(f.source_path, f.source, 100)
    f.study.raw["bases"][0]["metadata"]["source_checkpoint_sha256"] = hashlib.sha256(f.source_path.read_bytes()).hexdigest()
    f.study = StudySpec.from_mapping(f.study.raw)
    path = f.source_path.parents[1] / "config.json"
    config = json.loads(path.read_text())
    config["depth_sensor_inverse"] = True
    path.write_text(json.dumps(config))
    failure(f, "source interface mismatch depth_sensor_inverse")


def test_declared_config_progress_and_duplicate_discovery_fail(tmp_path):
    f = fixture(tmp_path)
    config_path = f.run_dir / "config.json"
    f.config["env"] = "wrong_task"
    config_path.write_text(json.dumps(f.config))
    failure(f, "config differs from StudySpec: env")
    f.config["env"] = "arbitrary_reward_task"
    config_path.write_text(json.dumps(f.config))
    _save(f.terminal, f.after, 32)
    failure(f, "checkpoint progress")
    (f.root / "duplicate" / "00_TRANSFER_S42").mkdir(parents=True)
    report = result(f)
    assert not report["passed"] and "exactly one directory" in report["errors"][0]


def test_softmax_logit_gauge_motion_does_not_qualify_as_cue_learning(tmp_path):
    f = fixture(tmp_path)
    f.config["fixed_task_goal_mixture"] = True
    (f.run_dir / "config.json").write_text(json.dumps(f.config))
    args = f.study.raw["training"]["common_args"]
    args[args.index("--fixed_task_goal_mixture=False")] = "--fixed_task_goal_mixture=True"
    f.study = StudySpec.from_mapping(f.study.raw)
    f.before[TASK] = torch.zeros(2, 3)
    f.after[TASK] = torch.ones(2, 3)
    _save(f.initial, f.before, 0)
    _save(f.terminal, f.after, 64)
    failure(f, "effective task condition did not move")


def test_cli_writes_provenance_and_exit_status(tmp_path, monkeypatch, capsys):
    from hpc_runs.intrmotiv_study.transfer_audit import main

    f = fixture(tmp_path)
    path, output = tmp_path / "study.json", tmp_path / "audit.json"
    path.write_text(json.dumps(f.study.raw))
    monkeypatch.setattr("sys.argv", ["transfer_audit", str(path), str(f.root), "--output", str(output)])
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 0
    assert json.loads(output.read_text())["passed"]
    assert json.loads(capsys.readouterr().out)["study_sha256"] == f.study.fingerprint
