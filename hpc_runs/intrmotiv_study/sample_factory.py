"""Thin adapter from StudySpec to Sample Factory's RunDescription."""

from __future__ import annotations

import shlex
from typing import Any, Callable

from .spec import RunSpec, StudySpec


def build_run_description(
    study: StudySpec,
    experiment_builder: Callable[[RunSpec], Any] | None = None,
) -> Any:
    """Build a RunDescription directly or through an explicit base adapter.

    ``experiment_builder`` is the extension point for established Python base
    factories that cannot yet be represented as complete declarative CLI
    bundles. The StudySpec still owns matrix expansion and run identities.
    """

    if study.training_mode != "sample_factory" and experiment_builder is None:
        raise RuntimeError(
            f"study {study.study_id!r} uses training.mode={study.training_mode!r}; "
            "it is not a complete Sample Factory run description"
        )
    try:
        from sample_factory.launcher.run_description import Experiment, RunDescription
    except ImportError as error:
        raise RuntimeError("Sample Factory is required to build a training RunDescription") from error
    if experiment_builder is None:
        experiments = [
            Experiment(run.name, " ".join(shlex.quote(arg) for arg in run.args), [{}])
            for run in study.expand_runs()
        ]
    else:
        experiments = [experiment_builder(run) for run in study.expand_runs()]
        observed_names = [experiment.name for experiment in experiments]
        expected_names = [run.name for run in study.expand_runs()]
        if observed_names != expected_names:
            raise RuntimeError(
                "custom experiment_builder must preserve StudySpec run names and order"
            )
    return RunDescription(study.batch_name, experiments=experiments)
