"""TensorBoard collection for standardized online study analysis."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


def collect_online_records(
    study: StudySpec,
    batch_root: Path,
    fixed_window: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Collect standardized per-run rows from TensorBoard event directories."""

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise RuntimeError("TensorBoard is required for collect-online") from error

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

    def collect_run(run: Any) -> dict[str, Any]:
        run_dir = run_directories[run.name]
        summary_dir = run_dir / ".summary" / "0"
        if not summary_dir.is_dir():
            raise SpecError(f"missing TensorBoard summary directory: {summary_dir}")
        accumulator = EventAccumulator(
            str(summary_dir), size_guidance={"scalars": scalar_size_guidance}
        )
        accumulator.Reload()
        available = set(accumulator.Tags().get("scalars", []))
        if step_tag not in available:
            raise SpecError(f"run {run.name!r} is missing step tag {step_tag!r}")
        max_step = max(int(event.step) for event in accumulator.Scalars(step_tag))
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
            if tag not in available:
                row[metric], row[f"{metric}__n"] = math.nan, 0
            else:
                row[metric], row[f"{metric}__n"] = mean_in_window(
                    accumulator.Scalars(tag), low, high
                )
        for metric, tag in cumulative_metrics.items():
            if tag not in available:
                row[metric], row[f"{metric}__step"] = math.nan, 0
            else:
                row[metric], row[f"{metric}__step"] = latest_at_or_before(
                    accumulator.Scalars(tag), high
                )
        return row

    runs = study.expand_runs()
    with ThreadPoolExecutor(max_workers=min(max_workers, len(runs))) as executor:
        # executor.map preserves StudySpec order while loading independent event
        # directories concurrently.
        return list(executor.map(collect_run, runs))
