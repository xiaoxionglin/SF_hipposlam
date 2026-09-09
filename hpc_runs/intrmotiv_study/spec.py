"""Validated declarative study specifications.

The specification owns scientific matrix metadata. Execution remains delegated
to Sample Factory for training and to the established manifest-driven evaluator
for place-field telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Mapping, Sequence

from .version import SCHEMA_ID, WORKFLOW_VERSION


Scalar = str | int | float | bool
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SEMVER_RE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


class SpecError(ValueError):
    """Raised when a study specification violates the v1 contract."""


def _mapping(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SpecError(f"{where} must be an object")
    return dict(value)


def _sequence(value: Any, where: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SpecError(f"{where} must be an array")
    return list(value)


def _identifier(value: Any, where: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise SpecError(f"{where} must match {_ID_RE.pattern!r}")
    return value


def _strings(value: Any, where: str) -> tuple[str, ...]:
    values = _sequence(value, where)
    if not all(isinstance(item, str) for item in values):
        raise SpecError(f"{where} must contain only strings")
    return tuple(values)


def _metadata(value: Any, where: str) -> dict[str, Scalar]:
    result = _mapping(value, where)
    for key, item in result.items():
        _identifier(key, f"{where} key")
        if not isinstance(item, (str, int, float, bool)):
            raise SpecError(f"{where}.{key} must be a scalar")
    return result


def _render(template: str, context: Mapping[str, Any], where: str) -> str:
    try:
        rendered = template.format_map(dict(context))
    except KeyError as error:
        raise SpecError(f"{where} references unknown field {error.args[0]!r}") from error
    if not rendered:
        raise SpecError(f"{where} rendered an empty string")
    return rendered


def _workspace_path(path: str, root: str, where: str) -> None:
    candidate = PurePosixPath(path)
    workspace = PurePosixPath(root)
    if not candidate.is_absolute() or not workspace.is_absolute():
        raise SpecError(f"{where} and workspace_root must be absolute")
    try:
        candidate.relative_to(workspace)
    except ValueError as error:
        raise SpecError(f"{where}={path!r} is outside workspace_root={root!r}") from error


def _semver(value: Any, where: str) -> tuple[int, int, int]:
    if not isinstance(value, str):
        raise SpecError(f"{where} must be a semantic version")
    match = _SEMVER_RE.fullmatch(value)
    if match is None:
        raise SpecError(f"{where} must be a semantic version")
    return tuple(int(item) for item in match.groups())


@dataclass(frozen=True)
class FactorLevel:
    value: Scalar
    label: str
    args: tuple[str, ...]
    metadata: Mapping[str, Scalar]


@dataclass(frozen=True)
class FactorSpec:
    name: str
    levels: tuple[FactorLevel, ...]


@dataclass(frozen=True)
class BaseSpec:
    name: str
    label: str
    args: tuple[str, ...]
    metadata: Mapping[str, Scalar]


@dataclass(frozen=True)
class RunSpec:
    name: str
    condition: str
    batch_name: str
    base: str
    seed: int
    factors: Mapping[str, Scalar]
    factor_labels: Mapping[str, str]
    metadata: Mapping[str, Scalar]
    args: tuple[str, ...]
    context: Mapping[str, Scalar]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "condition": self.condition,
            "batch": self.batch_name,
            "base": self.base,
            "seed": self.seed,
            "factors": dict(self.factors),
            "factor_labels": dict(self.factor_labels),
            "metadata": dict(self.metadata),
            "args": list(self.args),
        }


@dataclass(frozen=True)
class StudySpec:
    source: Path
    raw: Mapping[str, Any]
    study_id: str
    description: str
    declared_workflow_version: str
    batch_name: str
    training_mode: str
    workspace_root: str
    output_root: str
    bases: tuple[BaseSpec, ...]
    factors: tuple[FactorSpec, ...]
    seeds: tuple[int, ...]
    common_args: tuple[str, ...]
    seed_arg: str
    run_name_template: str
    condition_name_template: str
    expected_runs: int
    analysis: Mapping[str, Any]
    telemetry: Mapping[str, Any]
    study_metadata: Mapping[str, Scalar]

    @classmethod
    def from_mapping(cls, raw_value: Mapping[str, Any], source: Path | None = None) -> "StudySpec":
        raw = dict(raw_value)
        if raw.get("schema") != SCHEMA_ID:
            raise SpecError(f"schema must be {SCHEMA_ID!r}")
        declared_version = raw.get("workflow_version")
        declared_semver = _semver(declared_version, "workflow_version")
        current_semver = _semver(WORKFLOW_VERSION, "current workflow version")
        if declared_semver[0] != current_semver[0] or declared_semver > current_semver:
            raise SpecError(
                f"workflow_version {declared_version!r} is not compatible with "
                f"installed version {WORKFLOW_VERSION!r}"
            )
        study_id = _identifier(raw.get("study_id"), "study_id")
        description = raw.get("description", "")
        if not isinstance(description, str) or not description.strip():
            raise SpecError("description must be a nonempty string")

        training = _mapping(raw.get("training"), "training")
        batch_name = _identifier(training.get("batch_name"), "training.batch_name")
        training_mode = training.get("mode", "sample_factory")
        if training_mode not in {"sample_factory", "supplemental_args"}:
            raise SpecError("training.mode must be 'sample_factory' or 'supplemental_args'")
        workspace_root = training.get("workspace_root")
        output_root = training.get("output_root")
        if not isinstance(workspace_root, str) or not isinstance(output_root, str):
            raise SpecError("training workspace_root and output_root must be strings")
        _workspace_path(output_root, workspace_root, "training.output_root")

        common_args = _strings(training.get("common_args", []), "training.common_args")
        seed_arg = training.get("seed_arg", "--seed={seed}")
        run_name_template = training.get("run_name_template")
        condition_template = training.get("condition_name_template", run_name_template)
        if not all(isinstance(item, str) for item in (seed_arg, run_name_template, condition_template)):
            raise SpecError("training seed_arg and name templates must be strings")

        bases: list[BaseSpec] = []
        for index, item_value in enumerate(_sequence(raw.get("bases"), "bases")):
            item = _mapping(item_value, f"bases[{index}]")
            name = _identifier(item.get("name"), f"bases[{index}].name")
            label = item.get("label", name)
            if not isinstance(label, str) or not label:
                raise SpecError(f"bases[{index}].label must be a nonempty string")
            bases.append(BaseSpec(
                name=name,
                label=label,
                args=_strings(item.get("args", []), f"bases[{index}].args"),
                metadata=_metadata(item.get("metadata", {}), f"bases[{index}].metadata"),
            ))
        if not bases or len({base.name for base in bases}) != len(bases):
            raise SpecError("bases must be nonempty and have unique names")

        factors: list[FactorSpec] = []
        for factor_index, factor_value in enumerate(_sequence(raw.get("factors", []), "factors")):
            factor = _mapping(factor_value, f"factors[{factor_index}]")
            name = _identifier(factor.get("name"), f"factors[{factor_index}].name")
            levels: list[FactorLevel] = []
            for level_index, level_value in enumerate(
                _sequence(factor.get("levels"), f"factors[{factor_index}].levels")
            ):
                level = _mapping(level_value, f"factors[{factor_index}].levels[{level_index}]")
                value = level.get("value")
                if not isinstance(value, (str, int, float, bool)):
                    raise SpecError(f"factor {name!r} level value must be a scalar")
                label = level.get("label", str(value))
                if not isinstance(label, str) or not label:
                    raise SpecError(f"factor {name!r} level label must be nonempty")
                levels.append(FactorLevel(
                    value=value,
                    label=label,
                    args=_strings(level.get("args", []), f"factor {name!r} level args"),
                    metadata=_metadata(level.get("metadata", {}), f"factor {name!r} level metadata"),
                ))
            if not levels or len({json.dumps(level.value, sort_keys=True) for level in levels}) != len(levels):
                raise SpecError(f"factor {name!r} must have nonempty, unique values")
            factors.append(FactorSpec(name=name, levels=tuple(levels)))
        if len({factor.name for factor in factors}) != len(factors):
            raise SpecError("factor names must be unique")

        seeds_value = _sequence(raw.get("seeds"), "seeds")
        if not seeds_value or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds_value):
            raise SpecError("seeds must be a nonempty array of integers")
        seeds = tuple(seeds_value)
        if len(set(seeds)) != len(seeds):
            raise SpecError("seeds must be unique")

        natural_count = len(bases) * len(seeds)
        for factor in factors:
            natural_count *= len(factor.levels)
        expected_runs = raw.get("expected_runs", natural_count)
        if expected_runs != natural_count:
            raise SpecError(
                f"expected_runs={expected_runs!r}, but the Cartesian product contains {natural_count} runs"
            )

        spec = cls(
            source=source or Path("<memory>"),
            raw=raw,
            study_id=study_id,
            description=description.strip(),
            declared_workflow_version=declared_version,
            batch_name=batch_name,
            training_mode=training_mode,
            workspace_root=workspace_root,
            output_root=output_root,
            bases=tuple(bases),
            factors=tuple(factors),
            seeds=seeds,
            common_args=common_args,
            seed_arg=seed_arg,
            run_name_template=run_name_template,
            condition_name_template=condition_template,
            expected_runs=expected_runs,
            analysis=_mapping(raw.get("analysis", {}), "analysis"),
            telemetry=_mapping(raw.get("telemetry", {}), "telemetry"),
            study_metadata=_metadata(raw.get("metadata", {}), "metadata"),
        )
        spec.expand_runs()  # Validate templates, argument rendering, and uniqueness now.
        return spec

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return sha256(canonical.encode("utf-8")).hexdigest()

    def expand_runs(self) -> list[RunSpec]:
        runs: list[RunSpec] = []
        level_products = product(*(factor.levels for factor in self.factors)) if self.factors else [()]
        combinations = list(level_products)
        for base in self.bases:
            for selected_levels in combinations:
                factor_values = {
                    factor.name: level.value
                    for factor, level in zip(self.factors, selected_levels)
                }
                factor_labels = {
                    factor.name: level.label
                    for factor, level in zip(self.factors, selected_levels)
                }
                for seed in self.seeds:
                    context: dict[str, Scalar] = {
                        "study_id": self.study_id,
                        "batch_name": self.batch_name,
                        "base": base.name,
                        "base_label": base.label,
                        "seed": seed,
                    }
                    context.update({f"base_{key}": value for key, value in base.metadata.items()})
                    for factor, level in zip(self.factors, selected_levels):
                        context[factor.name] = level.value
                        context[f"{factor.name}_label"] = level.label
                        context.update({
                            f"{factor.name}_{key}": value for key, value in level.metadata.items()
                        })
                    metadata = dict(self.study_metadata)
                    metadata.update(base.metadata)
                    for factor, level in zip(self.factors, selected_levels):
                        metadata.update({f"{factor.name}_{key}": value for key, value in level.metadata.items()})
                    arg_templates = [
                        *self.common_args,
                        *base.args,
                        *(arg for level in selected_levels for arg in level.args),
                        self.seed_arg,
                    ]
                    args = tuple(
                        _render(template, context, "training argument") for template in arg_templates
                    )
                    if not all(arg.startswith("--") for arg in args):
                        raise SpecError("every rendered training argument must begin with '--'")
                    flags = [arg.split("=", 1)[0] for arg in args]
                    duplicate_flags = sorted({flag for flag in flags if flags.count(flag) > 1})
                    if duplicate_flags:
                        raise SpecError(
                            f"run {_render(self.run_name_template, context, 'run name')!r} "
                            f"defines duplicate flags {duplicate_flags!r}"
                        )
                    runs.append(RunSpec(
                        name=_render(self.run_name_template, context, "training.run_name_template"),
                        condition=_render(
                            self.condition_name_template, context, "training.condition_name_template"
                        ),
                        batch_name=self.batch_name,
                        base=base.name,
                        seed=seed,
                        factors=factor_values,
                        factor_labels=factor_labels,
                        metadata=metadata,
                        args=args,
                        context=context,
                    ))
        names = [run.name for run in runs]
        if len(runs) != self.expected_runs or len(set(names)) != len(names):
            raise SpecError("expanded runs must match expected_runs and have unique names")
        return runs

    def provenance(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_ID,
            "workflow_version": WORKFLOW_VERSION,
            "study_workflow_version": self.declared_workflow_version,
            "study_id": self.study_id,
            "study_spec": str(self.source),
            "study_sha256": self.fingerprint,
            "batch_name": self.batch_name,
            "training_mode": self.training_mode,
            "expected_runs": self.expected_runs,
        }


def load_study(path: str | Path) -> StudySpec:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SpecError(f"cannot load study specification {source}: {error}") from error
    return StudySpec.from_mapping(_mapping(raw, "study"), source=source)
