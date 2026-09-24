"""TensorBoard collection for standardized online study analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing
import csv
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping

from .discovery import discover_run_directories
from .spec import SpecError, StudySpec


def mean_in_window(events: Iterable[Any], low: int, high: int) -> tuple[float, int]:
    values = [float(event.value) for event in events if low <= int(event.step) <= high]
    return (fmean(values), len(values)) if values else (math.nan, 0)


def latest_at_or_before(events: Iterable[Any], high: int) -> tuple[float, int]:
    values = [event for event in events if int(event.step) <= high]
    if not values:
        return math.nan, 0
    latest = max(values, key=lambda event: (int(event.step), float(event.wall_time)))
    return float(latest.value), int(latest.step)


def _load_run_histories(run_name, run_dir, step_tag, required, scalar_size_guidance, latest_common):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    summary_dir = run_dir / ".summary" / "0"
    if not summary_dir.is_dir():
        raise SpecError(f"missing TensorBoard summary directory: {summary_dir}")
    accumulator = EventAccumulator(
        str(summary_dir), size_guidance={"scalars": 0 if latest_common else scalar_size_guidance}
    )
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    if step_tag not in available:
        raise SpecError(f"run {run_name!r} is missing step tag {step_tag!r}")
    histories = {tag: accumulator.Scalars(tag) for tag in required & available}
    if not histories[step_tag]:
        raise SpecError(f"run {run_name!r} has an empty step history")
    if latest_common:
        missing = [tag for tag in required if not histories.get(tag)]
        if missing:
            raise SpecError(f"run {run_name!r} is missing required histories: {missing}")
    return run_dir, histories


def collect_online_records(
    study: StudySpec,
    batch_root: Path,
    fixed_window: tuple[int, int] | None = None,
    latest_common: bool = False,
    progress=None,
    history_output_dir: Path | None = None,
    loader_backend: str | None = None,
) -> list[dict[str, Any]]:
    """Collect standardized per-run rows from TensorBoard event directories."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required for collect-online") from error

    if latest_common and fixed_window is not None:
        raise SpecError("latest-common and a fixed window are mutually exclusive")

    analysis = study.analysis
    window_metrics = analysis.get("window_metrics", {})
    cumulative_metrics = analysis.get("cumulative_metrics", {})
    if not isinstance(window_metrics, Mapping) or not isinstance(cumulative_metrics, Mapping):
        raise SpecError("analysis metric maps must be objects")
    step_tag = analysis.get("step_tag", "train/env_steps")
    terminal_width = analysis.get("terminal_width", 10_000_000)
    scalar_size_guidance = analysis.get("scalar_size_guidance", 30_000)
    max_workers = analysis.get("max_workers", 4)
    if (
        not isinstance(step_tag, str)
        or not isinstance(terminal_width, int)
        or terminal_width <= 0
        or not isinstance(scalar_size_guidance, int)
        or scalar_size_guidance <= 0
        or not isinstance(max_workers, int)
        or max_workers <= 0
    ):
        raise SpecError("analysis collection settings are invalid")

    run_directories = discover_run_directories(study, batch_root)

    runs = study.expand_runs()
    required = {step_tag, *window_metrics.values(), *cumulative_metrics.values()}
    backend = loader_backend or analysis.get("loader_backend", "thread")
    if backend not in ("thread", "process"):
        raise SpecError("analysis.loader_backend must be thread or process")
    executor_type = ProcessPoolExecutor if backend == "process" else ThreadPoolExecutor
    options = {"max_workers": min(max_workers, len(runs))}
    if backend == "process":
        # Spawn avoids inheriting TensorBoard/PyTorch background threads.
        options["mp_context"] = multiprocessing.get_context("spawn")
    loaded = [None] * len(runs)
    if history_output_dir is not None:
        history_output_dir.mkdir(parents=True, exist_ok=True)
    with executor_type(**options) as executor:
        futures = {
            executor.submit(_load_run_histories, run.name, run_directories[run.name], step_tag,
                            required, scalar_size_guidance, latest_common or history_output_dir is not None): index
            for index, run in enumerate(runs)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            loaded[index] = future.result()
            if history_output_dir is not None:
                run_dir, histories = loaded[index]
                # Retain every selected event, including repeated steps. Consumers
                # choose aggregation explicitly rather than inheriting a reservoir.
                name = runs[index].name
                with (history_output_dir / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["tag", "step", "wall_time", "value"])
                    for tag in sorted(histories):
                        writer.writerows((tag, e.step, e.wall_time, e.value) for e in histories[tag])
                sources = [{"path": str(p), "bytes": p.stat().st_size,
                            "mtime_ns": p.stat().st_mtime_ns}
                           for p in sorted((run_dir / ".summary" / "0").glob("events*"))]
                (history_output_dir / f"{name}.json").write_text(json.dumps(
                    {"run_name": name, "sources": sources, "all_selected_events": True}, indent=2) + "\n")
            if progress is not None:
                progress(completed, len(runs), runs[index].name)
    if latest_common:
        high = min(max(int(event.step) for event in events)
                   for _, histories in loaded for events in histories.values())
        low = max(0, high - terminal_width)
        if low >= high:
            raise SpecError("no positive latest-common window is available")
        fixed_window = (low, high)

    records = []
    for run, (run_dir, histories) in zip(runs, loaded):
        max_step = max(int(event.step) for event in histories[step_tag])
        if fixed_window is None:
            low, high = max(0, max_step - terminal_width), max_step
        else:
            low, high = fixed_window
            if low < 0 or low >= high or high > max_step:
                raise SpecError(f"run {run.name!r} cannot supply fixed window {low}--{high}")
        row: dict[str, Any] = {
            "run_name": run.name,
            "condition": run.condition,
            "base": run.base,
            "seed": run.seed,
            **run.factors,
            **run.metadata,
            "max_step": max_step,
            "window_low": low,
            "window_high": high,
            "run_dir": str(run_dir),
        }
        for metric, tag in window_metrics.items():
            if tag not in histories:
                row[metric], row[f"{metric}__n"] = math.nan, 0
            else:
                row[metric], row[f"{metric}__n"] = mean_in_window(
                    histories[tag], low, high
                )
        for metric, tag in cumulative_metrics.items():
            if tag not in histories:
                row[metric], row[f"{metric}__step"] = math.nan, 0
            else:
                row[metric], row[f"{metric}__step"] = latest_at_or_before(
                    histories[tag], high
                )
        if latest_common:
            for metric in window_metrics:
                if row[f"{metric}__n"] == 0 or not math.isfinite(row[metric]):
                    raise SpecError(f"run {run.name!r} has no finite {metric!r} mean in {low}--{high}")
        records.append(row)
    return records
