"""Shared study-level aggregation and paired linear contrasts."""

from __future__ import annotations

from collections import defaultdict
import math
from statistics import fmean, stdev
from typing import Any, Iterable, Mapping, Sequence

from .spec import SpecError


def _key(record: Mapping[str, Any], fields: Sequence[str]) -> tuple[Any, ...]:
    try:
        return tuple(record[field] for field in fields)
    except KeyError as error:
        raise SpecError(f"analysis record is missing grouping field {error.args[0]!r}") from error


def _number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise SpecError(f"metric {field!r} contains nonnumeric value {value!r}") from error
    return result


def summarize_records(
    records: Iterable[Mapping[str, Any]],
    group_by: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    """Return wide mean, sample-SD, and finite-count summaries."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[_key(record, group_by)].append(record)
    rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped, key=lambda value: tuple(map(str, value))):
        members = grouped[group_key]
        row = dict(zip(group_by, group_key))
        for metric in metrics:
            values = [_number(member.get(metric), metric) for member in members]
            finite = [value for value in values if math.isfinite(value)]
            row[f"{metric}__mean"] = fmean(finite) if finite else math.nan
            row[f"{metric}__sd"] = stdev(finite) if len(finite) > 1 else math.nan
            row[f"{metric}__n"] = len(finite)
        rows.append(row)
    return rows


def _matches(record: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    return all(record.get(field) == value for field, value in selector.items())


def linear_contrasts(
    records: Iterable[Mapping[str, Any]],
    metrics: Sequence[str],
    group_by: Sequence[str],
    replicate_by: Sequence[str],
    contrasts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate explicit within-replicate linear contrasts.

    Each contrast contains ``terms`` with ``weight`` and ``where``. A term must
    select exactly one row within every group/replicate cell. This makes paired
    scientific comparisons explicit and prevents silent averaging over an
    omitted factor.
    """

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    cell_fields = [*group_by, *replicate_by]
    for record in records:
        grouped[_key(record, cell_fields)].append(record)

    detailed: list[dict[str, Any]] = []
    for contrast in contrasts:
        name = contrast.get("name")
        if not isinstance(name, str) or not name:
            raise SpecError("every analysis contrast needs a nonempty name")
        terms = contrast.get("terms")
        if isinstance(terms, (str, bytes)) or not isinstance(terms, Sequence) or not terms:
            raise SpecError(f"contrast {name!r} needs a nonempty terms array")
        parsed_terms: list[tuple[float, Mapping[str, Any]]] = []
        for term in terms:
            if not isinstance(term, Mapping) or not isinstance(term.get("where"), Mapping):
                raise SpecError(f"contrast {name!r} terms need weight and where")
            parsed_terms.append((_number(term.get("weight"), "weight"), term["where"]))

        for cell_key, members in grouped.items():
            selected: list[tuple[float, Mapping[str, Any]]] = []
            for weight, selector in parsed_terms:
                matches = [member for member in members if _matches(member, selector)]
                if len(matches) != 1:
                    cell = dict(zip(cell_fields, cell_key))
                    raise SpecError(
                        f"contrast {name!r} term {dict(selector)!r} selected {len(matches)} "
                        f"rows in paired cell {cell!r}; expected exactly one"
                    )
                selected.append((weight, matches[0]))
            for metric in metrics:
                value = sum(weight * _number(record.get(metric), metric) for weight, record in selected)
                detailed.append({
                    **dict(zip(cell_fields, cell_key)),
                    "contrast": name,
                    "metric": metric,
                    "value": value,
                })

    summary_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    summary_fields = [*group_by, "contrast", "metric"]
    for row in detailed:
        value = float(row["value"])
        if math.isfinite(value):
            summary_groups[_key(row, summary_fields)].append(value)
    summary: list[dict[str, Any]] = []
    for group_key in sorted(summary_groups, key=lambda value: tuple(map(str, value))):
        values = summary_groups[group_key]
        summary.append({
            **dict(zip(summary_fields, group_key)),
            "mean": fmean(values),
            "sd": stdev(values) if len(values) > 1 else math.nan,
            "n": len(values),
        })
    return detailed, summary

