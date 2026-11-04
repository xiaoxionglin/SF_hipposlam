"""Real tiny checkpoint/event fixtures; no DMLab or external source required."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from hpc_runs.intrmotiv_study.spec import StudySpec
from hpc_runs.intrmotiv_study.transfer_audit import DG, GRADIENT_TAGS, TASK, audit


def _save(path, model, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    step = 0 if frames == 0 else 1
    moments = {
        "step": torch.tensor(float(step)),
        "exp_avg": torch.zeros_like(model[DG[0] + "weight"]),
        "exp_avg_sq": torch.zeros_like(model[DG[0] + "weight"]),
    }
    optimizer = {"state": {0: moments} if step else {}, "param_groups": [{"params": [0], "weight_decay": 0}]}
    torch.save({"model": model, "env_steps": frames, "train_step": step, "optimizer": optimizer}, path)


def _events(run_dir, *, reward=0.25, mismatch=False, omit=None, gradients=None):
    with SummaryWriter(str(run_dir / ".summary" / "0")) as writer:
        for step in (16, 32):
            scalars = {
                "train/env_steps": step,
                "intrmotiv/reward/advantage_mean": reward + float(mismatch),
                "intrmotiv/reward/environment_mean": reward,
                "train/policy_loss": 0.1,
            }
            scalars.update({GRADIENT_TAGS[name]: value for name, value in (gradients or {}).items()})
            for tag, value in scalars.items():
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
        "env_frameskip": 4,
        "core_name": "ExampleCA3",
        "Hippo_n_feature": 3,
        "Hippo_R": 2,
        "Hippo_L": 4,
        "encoder_conv_architecture": "example",
        "depth_sensor": True,
        "normalize_input": False,
        "hrl_goal_conditioning": "target_id_film",
        "dmlab_navigation_action_set": True,
    }
    (source_path.parents[1] / "config.json").write_text(json.dumps(source_config))
    config = {
        **source_config,
        "env": "arbitrary_reward_task",
        "depth_sensor_inverse": False,
        "transfer_scope": scope,
        "transfer_model_path": str(source_path),
        "transfer_freeze_dg": True,
        "transfer_freeze_worker": False,
        "advantage_reward_source": "external",
        "num_policies": 1,
        "with_pbt": False,
        "extra_encoder_losses": False,
        "fixed_task_conditioning": True,
        "fixed_task_goal_mixture": False,
        "reward_instruction_count": 2,
        "hrl_controllable_graph": False,
        "reward_scale": 0.1,
        "train_for_env_steps": 64,
        "seed": 42,
    }
    raw = {
        "schema": "intrmotiv/study/v1",
        "workflow_version": "1.14.1",
        "study_id": "transfer_fixture",
        "description": "Synthetic qualification audit fixtures",
        "training": {
            "batch_name": "batch",
            "workspace_root": str(tmp_path),
            "output_root": str(tmp_path / "batch"),
            "run_name_template": "{base}_S{seed}",
            "condition_name_template": "{base}",
            "common_args": [f"--{key}={value}" for key, value in config.items() if key != "seed"],
        },
        "bases": [
            {
                "name": "TRANSFER",
                "metadata": {
                    "source_checkpoint": str(source_path),
                    "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                },
            }
        ],
        "seeds": [42],
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
    return SimpleNamespace(
        study=study,
        root=tmp_path / "batch",
        config=config,
        run_dir=run_dir,
        before=before,
        after=after,
        initial=initial,
        terminal=terminal,
        source=source,
        source_path=source_path,
    )


def result(f):
    return audit(f.study, f.root)


def failure(f, message):
    report = result(f)
    assert not report["passed"]
    assert message in " ".join(report["runs"][0]["errors"])


def _configure(f, **updates):
    """Change the declared study and saved config together, as a real new cell."""
    f.config.update(updates)
    (f.run_dir / "config.json").write_text(json.dumps(f.config))
    f.study.raw["training"]["common_args"] = [f"--{key}={value}" for key, value in f.config.items() if key != "seed"]
    f.study = StudySpec.from_mapping(f.study.raw)


def _plastic_fixture(tmp_path, mode, scope="policy"):
    f = fixture(tmp_path, scope, events=False)
    ppo, encoder = {
        "bn_only": (False, False),
        "ppo_only": (True, False),
        "intrinsic_only": (False, True),
        "joint": (True, True),
    }[mode]
    _configure(
        f,
        transfer_freeze_dg=False,
        ppo_dg_gradient="joint" if ppo else "stop",
        encoder_grad_coeff=float(encoder),
        dg_batchnorm_semantics="legacy_batch",
    )
    if mode == "bn_only":
        weight = f.source[DG[0] + "weight"]
        weight.div_(weight.norm(dim=1, keepdim=True))
        for model in (f.before, f.after):
            model[DG[0] + "weight"] = weight.clone()
        _save(f.source_path, f.source, 100)
        f.study.raw["bases"][0]["metadata"]["source_checkpoint_sha256"] = hashlib.sha256(
            f.source_path.read_bytes()
        ).hexdigest()
        f.study = StudySpec.from_mapping(f.study.raw)
        _save(f.initial, f.before, 0)
    if ppo or encoder:
        f.after[DG[0] + "weight"] += 0.01
    f.after[DG[1] + "running_mean"] += 0.1
    f.after[DG[1] + "num_batches_tracked"] += 2
    _save(f.terminal, f.after, 64)
    _events(f.run_dir, gradients={"ppo": 0.2 if ppo else 0.0, "encoder": 0.3 if encoder else 0.0})
    return f


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


@pytest.mark.parametrize(
    "key",
    [
        "encoder.DG_projection.batchnorm1d.running_mean",
        "encoder.DG_projection.batchnorm1d.num_batches_tracked",
        "encoder.DG_projection.weight_generation",
    ],
)
def test_any_frozen_projection_buffer_drift_fails(tmp_path, key):
    f = fixture(tmp_path)
    f.after[key] += 1
    _save(f.terminal, f.after, 64)
    failure(f, "frozen DG parameters/BN/buffers changed")


@pytest.mark.parametrize(
    "change,message",
    [
        ("missing_bn", "missing compared tensor"),
        ("source_dg", "source DG"),
        ("worker_load", "worker transfer mismatch"),
        ("worker_stuck", "worker did not change"),
        ("critic_copy", "critic was not freshly initialized"),
        ("cue_stuck", "every cue"),
        ("nonfinite", "nonfinite task parameters"),
        ("cue_shape", "task parameter shape"),
    ],
)
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


@pytest.mark.parametrize(
    "variant,message",
    [
        ("missing", "missing TensorBoard"),
        ("zero", "positive external reward batches"),
        ("mismatch", "reward mismatch"),
        ("tag", "missing TensorBoard tags"),
        ("nan", "nonfinite learning scalar"),
    ],
)
def test_reward_data_fail_closed(tmp_path, variant, message):
    f = fixture(tmp_path, events=False)
    if variant != "missing":
        _events(
            f.run_dir,
            reward=0 if variant == "zero" else float("nan") if variant == "nan" else 0.25,
            mismatch=variant == "mismatch",
            omit="intrmotiv/reward/environment_mean" if variant == "tag" else None,
        )
    failure(f, message)


def test_hash_and_source_depth_interface_are_checked(tmp_path):
    f = fixture(tmp_path)
    with f.source_path.open("ab") as handle:
        handle.write(b"changed")
    failure(f, "SHA-256 mismatch")
    # Restore the hash, then alter only the source interface (not tensor shapes).
    _save(f.source_path, f.source, 100)
    f.study.raw["bases"][0]["metadata"]["source_checkpoint_sha256"] = hashlib.sha256(
        f.source_path.read_bytes()
    ).hexdigest()
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


@pytest.mark.parametrize("scope", ["policy", "dg"])
@pytest.mark.parametrize("mode", ["bn_only", "ppo_only", "intrinsic_only", "joint"])
def test_declared_plasticity_checks_weights_moments_and_each_gradient_route(tmp_path, scope, mode):
    f = _plastic_fixture(tmp_path, mode, scope)
    report = result(f)
    assert report["passed"], report
    row = report["runs"][0]
    assert row["dg_mode"] == mode
    assert bool(row["dg_weight_changed_tensors"]) == (mode != "bn_only")
    assert row["dg_bn_changed_tensors"]
    assert row["source_used_for_initialization"]
    assert set(row["dg_gradient_maxima"]) == {"ppo", "encoder"}


@pytest.mark.parametrize(
    "mode,wrong,message",
    [
        ("bn_only", {"ppo": 0.1, "encoder": 0.0}, "DG ppo gradient must be zero"),
        ("ppo_only", {"ppo": 0.0, "encoder": 0.0}, "DG ppo gradient must be nonzero"),
        ("ppo_only", {"ppo": 0.1, "encoder": 0.1}, "DG encoder gradient must be zero"),
        ("intrinsic_only", {"ppo": 0.0, "encoder": 0.0}, "DG encoder gradient must be nonzero"),
        ("joint", {"ppo": 0.0, "encoder": 0.1}, "DG ppo gradient must be nonzero"),
    ],
)
def test_gradient_ownership_mismatch_fails(tmp_path, mode, wrong, message):
    f = _plastic_fixture(tmp_path, mode)
    _events(f.run_dir, gradients=wrong)
    failure(f, message)


@pytest.mark.parametrize(
    "mode,mutation,message",
    [
        ("bn_only", "weight", "BN-only projection no longer has unit rows"),
        ("bn_only", "buffer", "BN-only adaptation changed non-BN buffers"),
        ("ppo_only", "weight_stuck", "DG projection weights did not change"),
        ("intrinsic_only", "moments_stuck", "normalization moments did not advance"),
        ("joint", "count_stuck", "normalization moments did not advance"),
    ],
)
def test_plasticity_tensor_contract_rejects_wrong_updates(tmp_path, mode, mutation, message):
    f = _plastic_fixture(tmp_path, mode)
    if mutation == "weight":
        f.after[DG[0] + "weight"] += 1
    elif mutation == "buffer":
        f.after["encoder.DG_projection.weight_generation"] += 1
    elif mutation == "weight_stuck":
        f.after[DG[0] + "weight"] = f.before[DG[0] + "weight"].clone()
    elif mutation == "moments_stuck":
        f.after[DG[1] + "running_mean"] = f.before[DG[1] + "running_mean"].clone()
    else:
        f.after[DG[1] + "num_batches_tracked"] = f.before[DG[1] + "num_batches_tracked"].clone()
    _save(f.terminal, f.after, 64)
    failure(f, message)


def test_bn_only_retained_unit_row_constraint_allows_float32_roundoff(tmp_path):
    f = _plastic_fixture(tmp_path, "bn_only")
    weight = f.after[DG[0] + "weight"]
    # Repeat exactly the post-optimizer operation; the audit does not require
    # bitwise agreement between CPU and CUDA reduction implementations.
    for _ in range(10):
        weight.div_(weight.norm(dim=1, keepdim=True).clamp_min(1e-6))
    _save(f.terminal, f.after, 64)
    report = result(f)
    assert report["passed"], report
    row = report["runs"][0]
    assert row["dg_constraint"] == "unit_row_normalization_roundoff"
    assert row["dg_optimizer_moments_exactly_zero"]
    assert 0 < row["dg_constraint_max_abs_delta"] < 1e-6


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("scale", "no longer has unit rows"),
        ("rotation", "beyond row-normalization roundoff"),
        ("first_rescaling", "did not start with unit rows"),
        ("first_moment", "exp_avg must be exactly zero"),
        ("second_moment", "exp_avg_sq must be exactly zero"),
        ("decay", "weight decay must be zero"),
        ("missing_state", "state is missing or ambiguous"),
        ("ambiguous_state", "state is missing or ambiguous"),
        ("stale_optimizer", "initial optimizer must be fresh"),
    ],
)
def test_bn_only_roundoff_exception_rejects_learning_and_real_transforms(tmp_path, mutation, message):
    f = _plastic_fixture(tmp_path, "bn_only")
    initial = torch.load(f.initial, weights_only=False)
    terminal = torch.load(f.terminal, weights_only=False)
    weight = terminal["model"][DG[0] + "weight"]
    if mutation == "scale":
        weight.mul_(1.001)
    elif mutation == "rotation":
        weight.copy_(weight.roll(1, dims=1))
    elif mutation == "first_rescaling":
        # Keep source transfer exact while testing a genuinely nonunit source.
        f.source[DG[0] + "weight"].mul_(2)
        initial["model"][DG[0] + "weight"].mul_(2)
        _save(f.source_path, f.source, 100)
        f.study.raw["bases"][0]["metadata"]["source_checkpoint_sha256"] = hashlib.sha256(
            f.source_path.read_bytes()
        ).hexdigest()
        f.study = StudySpec.from_mapping(f.study.raw)
    elif mutation in ("first_moment", "second_moment"):
        key = "exp_avg" if mutation == "first_moment" else "exp_avg_sq"
        terminal["optimizer"]["state"][0][key].fill_(1e-12)
    elif mutation == "decay":
        terminal["optimizer"]["param_groups"][0]["weight_decay"] = 1e-12
    elif mutation == "missing_state":
        terminal["optimizer"]["state"].clear()
    elif mutation == "ambiguous_state":
        terminal["optimizer"]["state"][1] = terminal["optimizer"]["state"][0].copy()
    else:
        initial["optimizer"] = terminal["optimizer"]
    torch.save(initial, f.initial)
    torch.save(terminal, f.terminal)
    failure(f, message)


def test_frozen_dg_still_rejects_even_one_float32_ulp(tmp_path):
    f = fixture(tmp_path)
    weight = f.after[DG[0] + "weight"]
    weight[0, 1] = torch.nextafter(weight[0, 1], torch.tensor(float("inf")))
    _save(f.terminal, f.after, 64)
    failure(f, "frozen DG parameters/BN/buffers changed")


def _scratch_fixture(tmp_path, calibrated):
    f = fixture(tmp_path, "dg", events=False)
    _configure(
        f,
        transfer_scope="none",
        transfer_model_path="",
        transfer_freeze_dg=calibrated,
        transfer_calibrate_frozen_dg=calibrated,
        encoder_grad_coeff=0.0,
        ppo_dg_gradient="stop" if calibrated else "joint",
    )
    f.before[DG[0] + "weight"] += 100
    f.before[DG[1] + "running_mean"].zero_()
    f.before[DG[1] + "running_var"].fill_(1)
    f.before[DG[1] + "num_batches_tracked"].zero_()
    for key in f.before:
        if key.startswith("encoder.DG_projection."):
            f.after[key] = f.before[key].clone()
    if not calibrated:
        f.after[DG[0] + "weight"] += 0.1
    f.after[DG[1] + "running_mean"] += 0.2
    f.after[DG[1] + "num_batches_tracked"] += 1 if calibrated else 2
    _save(f.initial, f.before, 0)
    _save(f.terminal, f.after, 64)
    _events(f.run_dir, gradients={"ppo": 0.0 if calibrated else 0.2, "encoder": 0.0})
    return f


@pytest.mark.parametrize("calibrated", [False, True])
def test_scratch_initialization_and_calibration_are_audited_without_transfer_path(tmp_path, calibrated):
    f = _scratch_fixture(tmp_path, calibrated)
    report = result(f)
    assert report["passed"], report
    row = report["runs"][0]
    assert not row["source_used_for_initialization"]
    assert row["source_checkpoint_sha256"] == hashlib.sha256(f.source_path.read_bytes()).hexdigest()
    assert row["dg_mode"] == ("calibrated_random_frozen" if calibrated else "ppo_only")


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("second_calibration", "calibrate exactly once"),
        ("weight", "weights or non-BN buffers changed"),
        ("buffer", "weights or non-BN buffers changed"),
        ("uncalibrated", "calibration did not change moments"),
        ("source_weights", "scratch DG projection copied source weights"),
        ("source_bn", "uncalibrated BN moments"),
    ],
)
def test_scratch_calibration_rejects_transfer_and_subsequent_drift(tmp_path, mutation, message):
    f = _scratch_fixture(tmp_path, True)
    if mutation == "second_calibration":
        f.after[DG[1] + "num_batches_tracked"] += 1
    elif mutation == "weight":
        f.after[DG[0] + "weight"] += 1
    elif mutation == "buffer":
        f.after["encoder.DG_projection.weight_generation"] += 1
    elif mutation == "uncalibrated":
        f.after[DG[1] + "running_mean"].zero_()
    elif mutation == "source_weights":
        f.before[DG[0] + "weight"] = f.source[DG[0] + "weight"].clone()
    else:
        f.before[DG[1] + "num_batches_tracked"] += 7
    _save(f.initial, f.before, 0)
    _save(f.terminal, f.after, 64)
    failure(f, message)


def test_frozen_source_worker_learns_only_critic_and_cues(tmp_path):
    f = fixture(tmp_path)
    _configure(f, transfer_freeze_worker=True)
    f.after["decoder.state.weight"] = f.before["decoder.state.weight"].clone()
    _save(f.terminal, f.after, 64)
    assert result(f)["passed"]
    f.after["action_parameterization.weight"] += 1
    _save(f.terminal, f.after, 64)
    failure(f, "frozen worker changed")


def test_plasticity_without_explicit_objective_flags_is_rejected(tmp_path):
    f = _plastic_fixture(tmp_path, "joint")
    f.study.raw["training"]["common_args"].remove("--encoder_grad_coeff=1.0")
    f.study = StudySpec.from_mapping(f.study.raw)
    failure(f, "explicitly declared freeze, PPO-gradient, and encoder-coefficient")


@pytest.mark.parametrize("name", ["gamma", "gae_lambda", "learning_rate", "encoder_grad_coeff", "ppo_dg_gradient"])
def test_new_declared_optimization_factors_cannot_drift(tmp_path, name):
    f = fixture(tmp_path)
    _configure(f, **{name: "stop" if name == "ppo_dg_gradient" else 0.1})
    f.config[name] = "joint" if name == "ppo_dg_gradient" else 0.2
    (f.run_dir / "config.json").write_text(json.dumps(f.config))
    failure(f, f"config differs from StudySpec: {name}")
