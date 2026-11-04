"""Fail-closed qualification of external-reward transfer and scratch controls.

Run with ``python -m hpc_runs.intrmotiv_study.transfer_audit STUDY BATCH_ROOT``.
Source checkpoints must have a metadata SHA-256 (source_checkpoint_sha256 or
source_sha256). This audits saved tensors and real learner telemetry, not task
performance. Initial checkpoints are mandatory; trained weights cannot certify
initialization. Only load trusted project checkpoints (PyTorch pickle format).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import re

import torch

from .discovery import discover_run_directories
from .spec import SpecError, load_study

DG = ("encoder.DG_projection.linear.", "encoder.DG_projection.batchnorm1d.")
WORKER = ("decoder.", "action_parameterization.", "core.dg_goal_modulation")
TASK = "core.fixed_task_target"
INTERFACE = ("env_frameskip", "core_name", "Hippo_n_feature", "Hippo_R", "Hippo_L",
             "encoder_conv_architecture", "depth_sensor", "normalize_input", "hrl_goal_conditioning")
DETECTOR_INTERFACE = ("DG_name", "DG_BN_intercept", "dg_batchnorm_semantics", "number_instruction_coef", "normalize_input_keys")
FALSE_DEFAULTS = ("depth_sensor_inverse", "dmlab_reduced_action_set",
                  "dmlab_extended_action_set", "dmlab_navigation_action_set")
REWARD_TAGS = ("intrmotiv/reward/advantage_mean", "intrmotiv/reward/environment_mean")
GRADIENT_TAGS = {name: f"intrmotiv/dg/gradient/{name}_norm" for name in ("ppo", "encoder")}


def _require(condition, message):
    if not condition:
        raise SpecError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload.get("model"), dict), f"missing model state dictionary: {path}")
    return payload


def _selected_checkpoints(run_dir):
    directory = run_dir / "checkpoint_p0"
    initial = list(directory.glob("initial_*.pth"))
    trained = []
    for path in directory.rglob("checkpoint_*.pth"):
        match = re.search(r"_(\d+)\.pth$", path.name)
        if match:
            frames = int(match.group(1))
            if frames == 0 and not initial:
                initial.append(path)  # Legacy frame-zero checkpoint naming.
            elif frames > 0:
                trained.append((frames, path))
    _require(len(initial) == 1, f"expected one initial checkpoint, found {len(initial)} in {directory}")
    _require(trained, f"missing trained checkpoint in {directory}")
    frames, terminal = max(trained)
    return initial[0], terminal, frames


def _changes(before, after, prefixes):
    keys = sorted({key for model in (before, after) for key in model if key.startswith(prefixes)})
    _require(keys, f"no tensors matching {prefixes}")
    for key in keys:
        _require(key in before and key in after, f"missing compared tensor {key}")
        _require(before[key].shape == after[key].shape, f"tensor shape mismatch: {key}")
        _require(torch.isfinite(before[key]).all() and torch.isfinite(after[key]).all(), f"nonfinite tensor {key}")
    return [key for key in keys if not torch.equal(before[key], after[key])]


def _interface_errors(source, destination):
    return [f"source interface mismatch {key}: {source.get(key)!r} vs {destination.get(key)!r}"
            for key in (*INTERFACE, *(key for key in DETECTOR_INTERFACE if key in source))
            if key not in source or source[key] != destination.get(key)] + [
        f"source interface mismatch {key}" for key in FALSE_DEFAULTS
        if source.get(key, False) != destination.get(key, False)]


def _source(config, metadata, cache):
    declared = metadata.get("source_checkpoint")
    # A scratch arm still names the source as a comparison reference, never as
    # an initializer. This lets the same audit reject accidentally copied DG,
    # worker, or critic tensors without adding a second initialization workflow.
    path = Path(declared or "") if config.get("transfer_scope") == "none" else Path(config.get("transfer_model_path", ""))
    _require(path.is_file(), f"missing source checkpoint: {path}")
    _require(not declared or path.resolve() == Path(declared).resolve(), "source path differs from StudySpec metadata")
    expected = metadata.get("source_checkpoint_sha256", metadata.get("source_sha256", ""))
    _require(re.fullmatch(r"[0-9a-fA-F]{64}", str(expected)), "missing valid source checkpoint SHA-256 in metadata")
    config_path = Path(metadata["source_config"]) if metadata.get("source_config") else next(
        (parent / "config.json" for parent in path.parents if (parent / "config.json").is_file()), None)
    _require(config_path and config_path.is_file(), f"missing source config.json for {path}")
    key = (str(path.resolve()), str(config_path.resolve()))
    if key not in cache:
        digest = _sha256(path)
        _require(digest == str(expected).lower(), f"source checkpoint SHA-256 mismatch: {path}")
        cache[key] = (digest, _checkpoint(path)["model"], json.loads(config_path.read_text()))
    digest, model, source_config = cache[key]
    _require(digest == str(expected).lower(), f"source checkpoint SHA-256 mismatch: {path}")
    errors = _interface_errors(source_config, config)
    _require(not errors, "; ".join(errors))
    return model, digest


def _event_audit(run_dir, min_positive_batches, gradient_expectations=None):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    directory = run_dir / ".summary" / "0"
    _require(directory.is_dir(), f"missing TensorBoard summary directory: {directory}")
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    tags = accumulator.Tags().get("scalars", [])
    required = (*REWARD_TAGS, "train/env_steps")
    if gradient_expectations is not None:
        required += tuple(GRADIENT_TAGS.values())
    _require(set(required).issubset(tags), f"missing TensorBoard tags: {sorted(set(required) - set(tags))}")
    histories = {}
    for tag in tags:
        if tag in required or "loss" in tag.lower() or "gradient" in tag.lower():
            events = accumulator.Scalars(tag)
            _require(all(math.isfinite(float(event.value)) for event in events), f"nonfinite learning scalar {tag}")
            # Resumed writers may repeat a step; keep its latest wall-time value.
            histories[tag] = {event.step: float(event.value) for event in sorted(events, key=lambda e: e.wall_time)}
    advantage, external = (histories[tag] for tag in REWARD_TAGS)
    _require(advantage and set(advantage) == set(external), "external/PPO reward histories have missing or unmatched steps")
    mismatch = [step for step in advantage if not math.isclose(advantage[step], external[step], abs_tol=1e-8, rel_tol=1e-6)]
    _require(not mismatch, f"external/PPO reward mismatch at steps {mismatch[:5]}")
    positives = sum(value > 0 for value in external.values())
    _require(positives >= min_positive_batches, f"positive external reward batches {positives} < {min_positive_batches}")
    max_frames = max(histories["train/env_steps"].values(), default=0)
    _require(max_frames > 0, "missing positive train/env_steps progress")
    gradient_maxima = {}
    if gradient_expectations is not None:
        for name, expected in gradient_expectations.items():
            values = histories[GRADIENT_TAGS[name]].values()
            _require(values and all(value >= 0 for value in values), f"invalid DG {name} gradient norms")
            maximum = max(values)
            _require(maximum > 0 if expected else maximum == 0,
                     f"DG {name} gradient must be {'nonzero' if expected else 'zero'}, observed maximum {maximum}")
            gradient_maxima[name] = maximum
    return {"reward_parity_batches": len(advantage), "positive_reward_batches": positives,
            "max_logged_frames": max_frames, "dg_gradient_maxima": gradient_maxima}


def _declared_config(run, config):
    """Check explicitly declared transfer/interface flags, not framework defaults."""
    names = {*INTERFACE, *DETECTOR_INTERFACE, *FALSE_DEFAULTS, "env", "transfer_scope", "transfer_model_path", "transfer_freeze_dg",
             "transfer_freeze_worker", "fixed_task_conditioning", "fixed_task_goal_mixture", "fixed_task_goal_init",
             "reward_instruction_count", "advantage_reward_source", "extra_encoder_losses", "hrl_controllable_graph",
             "reward_scale", "with_pbt", "num_policies", "seed", "train_for_env_steps",
             "transfer_calibrate_frozen_dg", "ppo_dg_gradient", "encoder_grad_coeff", "gamma", "gae_lambda",
             "learning_rate", "lr_schedule", "dg_goal_input", "dg_context_feedback", "dg_transition_prediction",
             "ca3_state_readout_mode", "dg_orthogonal_recruitment", "iterative_update", "distance_learning"}
    for argument in run.args:
        key, separator, value = argument.removeprefix("--").partition("=")
        if separator and key in names:
            actual = config.get(key)
            matches = str(actual).lower() == value.lower()
            if isinstance(actual, (int, float)) and not isinstance(actual, bool):
                try:
                    matches = float(value) == actual
                except ValueError:
                    matches = False
            _require(matches, f"config differs from StudySpec: {key}={actual!r}, expected {value!r}")


def _dg_contract(config, run):
    """Derive only explicitly declared plasticity controls; legacy means frozen."""
    frozen = config.get("transfer_freeze_dg")
    _require(isinstance(frozen, bool), "qualification requires an explicit transfer_freeze_dg boolean")
    calibration = config.get("transfer_calibrate_frozen_dg", False)
    scope = config.get("transfer_scope")
    _require(scope in ("none", "dg", "policy"), f"unsupported transfer_scope {scope!r}; expected none, dg or policy")
    _require(bool(config.get("transfer_model_path")) == (scope != "none"),
             "transfer_model_path and a non-none transfer_scope must be supplied together")
    _require(not calibration or (frozen and scope == "none"), "calibration requires a fresh frozen DG")
    _require(not config.get("transfer_freeze_worker", False) or scope == "policy",
             "a frozen worker requires policy transfer")
    declared = {argument.removeprefix("--").partition("=")[0] for argument in run.args}
    if not frozen:
        _require({"transfer_freeze_dg", "ppo_dg_gradient", "encoder_grad_coeff"}.issubset(declared),
                 "plastic DG qualification requires explicitly declared freeze, PPO-gradient, and encoder-coefficient flags")
        _require(config.get("dg_batchnorm_semantics", "legacy_batch") == "legacy_batch",
                 "plastic DG qualification currently supports legacy_batch normalization")
        _require(config.get("distance_learning", True) and not config.get("iterative_update", False),
                 "plastic DG qualification requires simultaneous DistanceLearner updates")
        for key, value in {"dg_goal_input": "none", "dg_context_feedback": "none",
                           "dg_transition_prediction": "none", "ca3_state_readout_mode": "off",
                           "dg_orthogonal_recruitment": False}.items():
            _require(config.get(key, value) == value, f"plastic DG qualification requires {key}={value!r}")
    gradient = config.get("ppo_dg_gradient", "stop")
    coefficient = float(config.get("encoder_grad_coeff", 1.0))
    _require(gradient in ("stop", "joint") and math.isfinite(coefficient) and coefficient >= 0,
             "invalid DG gradient configuration")
    expected = {"ppo": not frozen and gradient == "joint", "encoder": not frozen and coefficient > 0}
    mode = ("calibrated_random_frozen" if calibration else "frozen") if frozen else (
        "joint" if all(expected.values()) else "ppo_only" if expected["ppo"] else
        "intrinsic_only" if expected["encoder"] else "bn_only")
    return mode, expected if not frozen or scope == "none" else None


def _require_fresh_bn(model):
    _require(model[DG[1] + "num_batches_tracked"].item() == 0 and
             torch.count_nonzero(model[DG[1] + "running_mean"]) == 0 and
             torch.equal(model[DG[1] + "running_var"], torch.ones_like(model[DG[1] + "running_var"])),
             "fresh random DG did not begin with uncalibrated BN moments")


def _audit_dg(before, after, mode):
    """Check weights separately from moments; calibration permits one BN update."""
    changes = _changes(before, after, ("encoder.DG_projection.",))
    weights = _changes(before, after, (DG[0],))
    bn = _changes(before, after, (DG[1],))
    count_key = DG[1] + "num_batches_tracked"
    if mode == "frozen":
        _require(not changes, "frozen DG parameters/BN/buffers changed during training")
    elif mode == "calibrated_random_frozen":
        allowed = {DG[1] + name for name in ("running_mean", "running_var", "num_batches_tracked")}
        _require(set(changes).issubset(allowed), "calibrated frozen DG weights or non-BN buffers changed")
        _require(before[count_key].item() == 0 and after[count_key].item() == 1,
                 "frozen random DG must calibrate exactly once (BN count 0 -> 1)")
        _require_fresh_bn(before)
        _require(any(key in bn for key in (DG[1] + "running_mean", DG[1] + "running_var")),
                 "frozen random DG calibration did not change moments")
    else:
        if mode == "bn_only":
            _require(not weights, "BN-only adaptation changed DG projection weights")
            _require(all(key.startswith(DG[1]) for key in changes), "BN-only adaptation changed non-BN buffers")
        else:
            _require(weights, f"{mode} DG projection weights did not change")
        _require(after[count_key].item() > before[count_key].item() and
                 any(key in bn for key in (DG[1] + "running_mean", DG[1] + "running_var")),
                 "plastic DG normalization moments did not advance")
    return {"dg_mode": mode, "dg_weight_changed_tensors": weights, "dg_bn_changed_tensors": bn,
            "dg_changed_tensors": changes}


def _audit_run(run, run_dir, metadata, cache, required_frames, min_positive_batches):
    config = json.loads((run_dir / "config.json").read_text())
    _declared_config(run, config)
    for key, value in {"advantage_reward_source": "external", "num_policies": 1,
                       "with_pbt": False, "extra_encoder_losses": False, "fixed_task_conditioning": True,
                       "hrl_controllable_graph": False}.items():
        _require(config.get(key) == value, f"qualification requires {key}={value!r}")
    scope = config.get("transfer_scope")
    dg_mode, gradient_expectations = _dg_contract(config, run)
    required = required_frames if required_frames is not None else int(config["train_for_env_steps"])
    _require(required > 0, "required frames must be positive")
    source, digest = _source(config, metadata, cache)
    initial_path, terminal_path, frames = _selected_checkpoints(run_dir)
    initial, terminal = _checkpoint(initial_path), _checkpoint(terminal_path)
    _require(initial.get("env_steps") == 0 and initial.get("train_step") == 0, "initial checkpoint is not before learning")
    _require(terminal.get("env_steps") == frames and frames >= required,
             f"checkpoint progress {terminal.get('env_steps')} (filename {frames}) below/inconsistent with {required}")
    before, after = initial["model"], terminal["model"]
    source_dg_changes = _changes(source, before, DG)
    if scope == "none":
        _require(any(key.startswith(DG[0]) for key in source_dg_changes), "scratch DG projection copied source weights")
        _require_fresh_bn(before)
    else:
        _require(not source_dg_changes, "initial source DG parameters/BN were not loaded exactly")
    dg_audit = _audit_dg(before, after, dg_mode)
    loaded_changes = _changes(source, before, WORKER)
    if scope == "policy":
        _require(not loaded_changes, f"initial worker transfer mismatch: {loaded_changes}")
    else:
        for prefix in ("decoder.", "action_parameterization."):
            _require(any(key.startswith(prefix) and key.endswith("weight") for key in loaded_changes),
                     f"fresh worker has no changed source weights in {prefix}")
    worker_changes = _changes(before, after, WORKER)
    frozen_worker = config.get("transfer_freeze_worker", False)
    _require(not worker_changes if frozen_worker else bool(worker_changes),
             "frozen worker changed" if frozen_worker else "worker did not change during learning")
    _changes(source, before, ("critic_linear.",))
    _require(_changes(before, after, ("critic_linear.",)), "critic did not change during learning")
    _require("critic_linear.weight" in before and not torch.equal(before["critic_linear.weight"], source["critic_linear.weight"]),
             "critic was not freshly initialized")
    _require(TASK in before and TASK in after, f"missing task parameter {TASK}")
    n_cues, n_dg = int(config.get("reward_instruction_count", 0)), int(config["Hippo_n_feature"])
    shape = (n_cues, n_dg) if n_cues else (n_dg,)
    _require(tuple(before[TASK].shape) == tuple(after[TASK].shape) == shape, f"task parameter shape must be {shape}")
    _require(torch.isfinite(before[TASK]).all() and torch.isfinite(after[TASK]).all(), "nonfinite task parameters")
    movement = (after[TASK] - before[TASK]).abs().reshape(max(1, n_cues), n_dg).sum(dim=1)
    _require((movement > 0).all(), f"task parameters did not move for every cue: {movement.tolist()}")
    effective = [model[TASK].softmax(-1) if config.get("fixed_task_goal_mixture") else model[TASK] for model in (before, after)]
    effective_movement = (effective[1] - effective[0]).abs().reshape(max(1, n_cues), n_dg).sum(dim=1)
    _require((effective_movement > 0).all(), "effective task condition did not move for every cue")
    events = _event_audit(run_dir, min_positive_batches, gradient_expectations)
    # Logging can legitimately end just before the final saved update.
    _require(events["max_logged_frames"] <= frames, "event progress exceeds latest checkpoint progress")
    return {"initial_checkpoint": str(initial_path), "terminal_checkpoint": str(terminal_path),
            "checkpoint_frames": frames, "required_frames": required, "source_checkpoint_sha256": digest,
            "source_used_for_initialization": scope != "none", **dg_audit,
            "worker_changed_tensors": worker_changes, "task_movement_l1_per_cue": movement.tolist(),
            "effective_task_movement_l1_per_cue": effective_movement.tolist(), **events}


def audit(study, batch_root, required_frames=None, min_positive_batches=1):
    """Audit every declared run; absent/ambiguous data always makes the report fail."""
    _require(min_positive_batches > 0, "min_positive_batches must be positive")
    result = {**study.provenance(), "protocol": "external-reward-transfer-qualification-v2", "passed": False, "runs": []}
    try:
        directories = discover_run_directories(study, Path(batch_root))
    except (OSError, SpecError) as error:
        return {**result, "errors": [str(error)]}
    cache = {}
    for run in study.expand_runs():
        row = {"run_name": run.name, "run_dir": str(directories[run.name]), "passed": False, "errors": []}
        try:
            row.update(_audit_run(run, directories[run.name], {**study.study_metadata, **run.metadata},
                                  cache, required_frames, min_positive_batches))
            row["passed"] = True
        except (OSError, ValueError, KeyError, RuntimeError, EOFError, TypeError, AttributeError, pickle.UnpicklingError) as error:
            row["errors"].append(f"{type(error).__name__}: {error}")
        result["runs"].append(row)
    result["passed_runs"] = sum(row["passed"] for row in result["runs"])
    result["passed"] = result["passed_runs"] == study.expected_runs
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--required-frames", type=int)
    parser.add_argument("--min-positive-batches", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(load_study(args.study), args.batch_root, args.required_frames, args.min_positive_batches)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
