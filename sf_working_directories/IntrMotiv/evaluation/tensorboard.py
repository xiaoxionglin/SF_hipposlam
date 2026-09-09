"""TensorBoard loading and terminal-window reduction shared by all reports."""

from __future__ import annotations

import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .schema import METRICS, measurement_scope, resolve_tag


def discover_event_files(batch_root: Path) -> list[Path]:
    """Return one newest event file per Sample Factory policy directory."""

    newest: dict[Path, Path] = {}
    for path in batch_root.rglob("events.out.tfevents.*"):
        if ".summary" not in path.parts:
            continue
        policy_dir = path.parents[2]
        if policy_dir not in newest or path.stat().st_mtime > newest[policy_dir].stat().st_mtime:
            newest[policy_dir] = path
    return sorted(newest.values())


def condition_metadata(run_name: str) -> dict[str, object]:
    normalized = run_name.removeprefix("00_")
    family, graph_scope = "flat", "none"
    if normalized.startswith(("GHRL_", "GAC_", "GAR_")):
        family, graph_scope = "global_fixed_hrl", "policy_global"
    elif normalized.startswith("LHRL_"):
        family, graph_scope = "stream_long_hrl", "per_stream"
    elif normalized.startswith("FLATLONG_"):
        family = "flat_long"
    elif "HRL" in normalized:
        family = "hrl"

    anti_collapse_arm = "none"
    if normalized.startswith(("FAC_", "GAC_")):
        anti_collapse_arm = "global_prethreshold"
    elif normalized.startswith(("FAR_", "GAR_")):
        anti_collapse_arm = "row_repulsion"

    def encoded_float(pattern: str) -> float:
        match = re.search(pattern, normalized)
        return float(match.group(1).replace("p", ".")) if match else math.nan

    seed = re.search(r"(?:^|_)S(\d+)(?:_|$)", normalized)
    half_life = re.search(r"(?:^|_)HL(\d+)(?:_|$)", normalized)
    update = "iterative" if "_iter_" in normalized else "simultaneous" if "_sim_" in normalized else "unknown"
    reward = re.search(r"_ER([A-Za-z0-9]+)_", normalized)
    return {
        "run": normalized,
        "family": family,
        "graph_scope": graph_scope,
        "seed": int(seed.group(1)) if seed else math.nan,
        "half_life_options": int(half_life.group(1)) if half_life else math.nan,
        "update_schedule": update,
        "encoder_reward_method": reward.group(1) if reward else "unknown",
        "dg_anti_collapse_arm": anti_collapse_arm,
        "dg_threshold": encoded_float(r"_T(\d+p\d+)_"),
        "dg_global_punishment_coeff": encoded_float(r"_G(\d+p\d+)_"),
        "dg_row_repulsion_coeff": encoded_float(r"_R(\d+p\d+)_"),
    }


def _mean_in_window(events, low: int, high: int) -> tuple[float, int]:
    values = [event.value for event in events if low <= event.step <= high]
    return (float(np.mean(values)), len(values)) if values else (math.nan, 0)


def parse_event_file(path: Path, batch_name: str, target_env_steps: int) -> dict[str, object]:
    # Long APPO jobs emit many unrelated scalars. A 2k cap is comfortably above
    # the terminal-window samples while keeping retrospective analysis bounded.
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 2_000})
    accumulator.Reload()
    available = accumulator.Tags().get("scalars", [])
    env_tag = resolve_tag(available, METRICS[0])
    env_events = accumulator.Scalars(env_tag) if env_tag else []
    max_step = max((event.step for event in env_events), default=0)
    if max_step == 0:
        max_step = max((event.step for tag in available for event in accumulator.Scalars(tag)), default=0)
    width = min(10_000_000, max(1_000_000, max_step // 5))
    row: dict[str, object] = {
        "batch": batch_name,
        "event_file": str(path),
        "max_step": max_step,
        "terminal_window_low": max(0, max_step - width),
        "terminal_window_high": max_step,
        "target_env_steps": target_env_steps,
        "training_progress": max_step / target_env_steps if target_env_steps else math.nan,
    }
    row.update(condition_metadata(path.parents[2].name))
    row["run_status"] = "complete" if row["training_progress"] >= 0.995 else "in_progress_or_short"
    for metric in METRICS:
        tag = resolve_tag(available, metric)
        row[f"{metric.key}__tag"] = tag or ""
        row[f"{metric.key}__scope"] = measurement_scope(tag)
        if not tag:
            row[metric.key], row[f"{metric.key}__samples"] = math.nan, 0
        else:
            row[metric.key], row[f"{metric.key}__samples"] = _mean_in_window(
                accumulator.Scalars(tag), row["terminal_window_low"], max_step
            )
    return row


def load_batches(batch_roots: list[Path], target_env_steps: int, workers: int = 1) -> list[dict[str, object]]:
    jobs = [
        (event_file, root.name, target_env_steps) for root in batch_roots for event_file in discover_event_files(root)
    ]
    if workers <= 1 or len(jobs) <= 1:
        return [parse_event_file(*job) for job in jobs]
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        return list(pool.map(_parse_job, jobs))


def _parse_job(job: tuple[Path, str, int]) -> dict[str, object]:
    return parse_event_file(*job)
