"""Release gates for the seven-arm predictive CA3 qualification batch."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from hpc_runs.audit_full_system_controller_preflight import audit as audit_controller
from hpc_runs.audit_navigation8_algorithm_screen_preflight import _events
from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict


READOUT_TAGS = (
    "intrmotiv/ca3_readout/enabled",
    "intrmotiv/ca3_readout/prediction_loss",
    "intrmotiv/ca3_readout/active_loss",
    "intrmotiv/ca3_readout/zero_loss",
    "intrmotiv/ca3_readout/valid_targets",
    "intrmotiv/ca3_readout/active_fraction",
    "intrmotiv/ca3_readout/state_shuffle_delta",
    "intrmotiv/ca3_readout/action_shuffle_delta",
)
CONTEXT_TAGS = (
    "active_goal_count",
    "anchor_registrations",
    "confirmation_attempts",
    "confirmation_successes",
    "anchor_replacements",
    "anchor_deactivations",
    "calibration_ready",
    "recognition_threshold",
    "prediction_absolute_threshold",
    "prediction_excess_threshold",
    "calibration_pair_count",
    "activation_latency_mean",
    "empty_set_exploration",
    "calibration_seconds",
)


def _latest(events, tag):
    values = events.get(tag, ())
    return float(max(values, key=lambda item: (item.step, item.wall_time)).value) if values else None


def _digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _certificates(root):
    if root is None:
        return {}
    records = {}
    for path in Path(root).glob("**/certificate.json"):
        row = json.loads(path.read_text())
        records[str(row["run"]).removeprefix("00_")] = row
    return records


def _model_buffer(model, suffix):
    matches = [value for key, value in model.items() if key.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"Expected one model buffer ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def audit(
    study,
    jobs,
    train_root,
    reload_certificate_root=None,
    telemetry_root=None,
    require_certified_terminal_successor=True,
):
    result = audit_controller(
        study,
        jobs,
        train_root,
        required_frames=2_000_000,
        require_certified_terminal_successor=require_certified_terminal_successor,
    )
    by_name = {row["run"]: row for row in result["runs"]}
    certificates = _certificates(reload_certificate_root)
    aliases = list(Path(telemetry_root).glob("**/contextual_alias_diagnostics.json")) if telemetry_root else []
    for job in csv.DictReader(Path(jobs).open(), delimiter="\t"):
        run = job["experiment"].removeprefix("00_")
        row = by_name[run]
        errors = row["errors"]
        directory = Path(train_root) / job["train_root"] / job["experiment"]
        cfg = json.loads((directory / "config.json").read_text())
        checkpoints = sorted((directory / "checkpoint_p0").glob("checkpoint_*.pth"))
        if not checkpoints:
            continue
        checkpoint = checkpoints[-1]
        payload = load_checkpoint_dict(checkpoint, torch.device("cpu"))
        events = _events(directory)

        metrics = {}
        for tag in READOUT_TAGS:
            values = events.get(tag, ())
            if not values:
                errors.append("missing CA3 batch scalar " + tag)
                continue
            if any(not math.isfinite(float(event.value)) for event in values):
                errors.append("nonfinite CA3 batch scalar " + tag)
            metrics[tag] = _latest(events, tag)
        readout_expected = cfg.get("ca3_state_readout_mode", "off") != "off"
        if metrics.get("intrmotiv/ca3_readout/enabled") != float(readout_expected):
            errors.append("readout enabled scalar disagrees with configuration")

        context_metrics = {}
        for name in CONTEXT_TAGS:
            tag = "intrmotiv/controller/" + name
            value = _latest(events, tag)
            if value is None or not math.isfinite(value):
                errors.append("missing or nonfinite contextual scalar " + tag)
            context_metrics[name] = value
        contextual = cfg.get("ca3_graph_anchor_mode", "off") != "off"
        model = payload["model"]
        if contextual:
            if (context_metrics.get("calibration_pair_count") or 0) < int(cfg["ca3_context_calibration_min_pairs"]):
                errors.append("contextual calibration did not collect the declared minimum pairs")
            if context_metrics.get("calibration_ready") != 1.0:
                errors.append("contextual calibration never became ready")
            if (context_metrics.get("anchor_registrations") or 0) <= 0:
                errors.append("anchor registration path was not exercised")
            if (context_metrics.get("confirmation_attempts") or 0) <= 0:
                errors.append("anchor confirmation path was not exercised")
        else:
            active = _model_buffer(model, "policy_graph.active_goal_mask")
            if bool(active.any()):
                errors.append("legacy/off arm acquired contextual active goals")

        certificate = certificates.get(run)
        if certificate is None:
            errors.append("missing exact checkpoint reload certificate")
        else:
            for field in ("exact_restore", "model_and_buffers_exact", "optimizer_exact", "counters_exact"):
                if certificate.get(field) is not True:
                    errors.append("reload certificate failed " + field)
            if certificate.get("checkpoint_sha256") != _digest(checkpoint):
                errors.append("reload certificate checkpoint digest mismatch")
            if int(certificate.get("frames", -1)) < 2_000_000:
                errors.append("reload certificate predates the qualification horizon")

        requires_alias = bool(cfg.get("ca3_graph_contextual_hits", False))
        if requires_alias and telemetry_root:
            matched = [path for path in aliases if run in str(path)]
            if not matched:
                errors.append("missing contextual alias telemetry")
            else:
                for path in matched:
                    diagnostic = json.loads(path.read_text())
                    if diagnostic.get("privileged_evaluation_only") is not True:
                        errors.append("contextual alias artifact lacks evaluation-only provenance")
        row["ca3_batch"] = dict(metrics=metrics, contextual=context_metrics, certificate=certificate)
        row["passed"] = not errors
    result["passed"] = all(row["passed"] for row in result["runs"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study")
    parser.add_argument("jobs")
    parser.add_argument("train_root")
    parser.add_argument("--reload-certificate-root", required=True)
    parser.add_argument("--telemetry-root", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(
        args.study,
        args.jobs,
        args.train_root,
        args.reload_certificate_root,
        args.telemetry_root,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    raise SystemExit(0 if result["passed"] else 1)
