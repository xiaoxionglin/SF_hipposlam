"""Physical destination arrivals from matched commands and independent raw maps.

Physical destinations are fixed before examining intervention trajectories.
The peak bin is deliberately distinct from the potentially broad canonical
half-peak component. Both are reported; neither depends on DG reward events.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .qualify_absent_goal_interventions import canonical_components


def bin_indices(points, shape, bounds=(100.0, 2000.0, 100.0, 2000.0)):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    lo, hi = np.array(bounds)[[0, 2]], np.array(bounds)[[1, 3]]
    valid = np.isfinite(points).all(1) & (points >= lo).all(1) & (points <= hi).all(1)
    indices = np.floor((np.nan_to_num(points) - lo) / (hi - lo) * shape).astype(int)
    return np.clip(indices, 0, np.array(shape) - 1), valid


def destination_arrival(row, mask, horizon):
    start = np.asarray(row["start_position"][:2], float)
    trajectory = np.asarray(row["trajectory"], float).reshape(-1, 2)[:horizon]
    indices, valid = bin_indices(np.vstack([start, trajectory]), mask.shape)
    inside = valid & mask[indices[:, 0], indices[:, 1]]
    result = dict(
        start_inside=bool(inside[0]),
        observed_decisions=len(trajectory),
        hit=False,
        latency=None,
        arrival_x=None,
        arrival_y=None,
        path_to_arrival=None,
        straight_line_efficiency=None,
    )
    # Already occupying a destination cannot be counted as commanded arrival.
    if inside[0] or not inside[1:].any():
        return result
    index = int(np.flatnonzero(inside[1:])[0])
    path = np.vstack([start, trajectory[: index + 1]])
    distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    direct = float(np.linalg.norm(path[-1] - start))
    result.update(
        hit=True,
        latency=index + 1,
        arrival_x=float(path[-1, 0]),
        arrival_y=float(path[-1, 1]),
        path_to_arrival=distance,
        straight_line_efficiency=direct / distance if distance > 0 else None,
    )
    return result


def evaluate_destinations(rows, masks, horizons):
    """Return trial-target records and start-paired, identity-macro contrasts."""
    records, contrasts = [], []
    for horizon in horizons:
        for start in sorted({r["start"] for r in rows}):
            trials = [r for r in rows if r["start"] == start]
            if not trials:
                continue
            commands = {r["command"] for r in trials}
            # Keep common observation time, complete command sets, and no
            # boundary-conditioned exclusion based on which command succeeded.
            if len(trials) != trials[0]["eligible_count"] or len(commands) != len(trials):
                continue
            if any(len(r["trajectory"]) < horizon for r in trials):
                continue
            for target in range(masks.shape[-1]):
                if target not in commands or len(commands) < 2 or not masks[..., target].any():
                    continue
                outcomes = [destination_arrival(r, masks[..., target], horizon) for r in trials]
                if any(o["start_inside"] for o in outcomes):
                    continue
                for trial, outcome in zip(trials, outcomes):
                    records.append(
                        dict(
                            start=start,
                            command=trial["command"],
                            target=target,
                            matched=trial["command"] == target,
                            horizon=horizon,
                            **outcome,
                        )
                    )
                matched = outcomes[next(i for i, r in enumerate(trials) if r["command"] == target)]
                other = [o for r, o in zip(trials, outcomes) if r["command"] != target]
                hit_other = float(np.mean([o["hit"] for o in other]))

                # Restricted latency includes misses as horizon+1, avoiding
                # success-only latency comparisons between unequal hit rates.
                def latency(outcome):
                    return outcome["latency"] if outcome["hit"] else horizon + 1

                contrasts.append(
                    dict(
                        start=start,
                        target=target,
                        horizon=horizon,
                        matched_hit=float(matched["hit"]),
                        mismatched_hit=hit_other,
                        hit_difference=float(matched["hit"]) - hit_other,
                        matched_restricted_latency=latency(matched),
                        mismatched_restricted_latency=float(np.mean([latency(o) for o in other])),
                        restricted_latency_difference=latency(matched) - float(np.mean([latency(o) for o in other])),
                    )
                )
    summary = {}
    for horizon in horizons:
        group = [r for r in contrasts if r["horizon"] == horizon]
        targets = sorted({r["target"] for r in group})
        means = [
            {
                key: float(np.mean([r[key] for r in group if r["target"] == t]))
                for key in (
                    "matched_hit",
                    "mismatched_hit",
                    "hit_difference",
                    "matched_restricted_latency",
                    "mismatched_restricted_latency",
                    "restricted_latency_difference",
                )
            }
            for t in targets
        ]
        summary[str(horizon)] = dict(
            total_identities=masks.shape[-1],
            represented_identities=len(targets),
            represented_target_ids=targets,
            paired_start_target_count=len(group),
            **{
                key: float(np.mean([r[key] for r in means])) if means else None
                for key in (
                    "matched_hit",
                    "mismatched_hit",
                    "hit_difference",
                    "matched_restricted_latency",
                    "mismatched_restricted_latency",
                    "restricted_latency_difference",
                )
            },
        )
        total = masks.shape[-1]
        missing = total - len(targets)
        matched_sum = sum(r["matched_hit"] for r in means)
        delta_sum = sum(r["hit_difference"] for r in means)
        summary[str(horizon)].update(
            all_identity_matched_hit=matched_sum / total if missing == 0 else None,
            all_identity_hit_difference=delta_sum / total if missing == 0 else None,
            all_identity_matched_hit_bounds=[matched_sum / total, (matched_sum + missing) / total],
            all_identity_hit_difference_bounds=[(delta_sum - missing) / total, (delta_sum + missing) / total],
            denominator_note="Point means above describe represented identities only. Full-identity bounds retain all identities and assign missing outcomes their possible range; these are missing-data bounds, not confidence intervals.",
        )
    return records, contrasts, summary


def trajectory_command_effect(rows):
    """Matched randomness isolates effects of commands on physical trajectories."""
    endpoints = []
    identical = []
    divergence = []
    for start in sorted({r["start"] for r in rows}):
        trials = [r for r in rows if r["start"] == start]
        for i, first in enumerate(trials):
            for second in trials[i + 1 :]:
                a = np.asarray(first["trajectory"], float).reshape(-1, 2)
                b = np.asarray(second["trajectory"], float).reshape(-1, 2)
                if len(a) != len(b) or len(a) == 0:
                    continue
                unequal = np.any(a != b, axis=1)
                identical.append(not unequal.any())
                endpoints.append(float(np.linalg.norm(a[-1] - b[-1])))
                if unequal.any():
                    divergence.append(int(np.flatnonzero(unequal)[0]) + 1)
    return dict(
        paired_command_pairs=len(identical),
        identical_trajectory_fraction=float(np.mean(identical)) if identical else None,
        endpoint_distance_mean=float(np.mean(endpoints)) if endpoints else None,
        first_physical_divergence_median=float(np.median(divergence)) if divergence else None,
    )


def analyze(trials, fields, out, map_key="raw_dg_rate_maps"):
    if map_key not in ("raw_dg_rate_maps", "rate_maps"):
        raise ValueError("Unsupported destination map basis")
    provenance = json.loads(trials.with_name("intervention_summary.json").read_text())
    if not provenance.get("exact_start_verified") or not provenance.get("policy_frozen"):
        raise ValueError("Physical command comparisons require verified identical frozen starts")
    frame = pd.read_csv(trials)
    rows = frame.to_dict("records")
    for row in rows:
        for key in ("trajectory", "start_position", "eligible_commands"):
            row[key] = json.loads(row[key])
    with np.load(fields, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    if Path(str(data["checkpoint"])).resolve() != Path(provenance["checkpoint"]).resolve():
        raise ValueError("Interventions and fields must use the identical checkpoint")
    if not str(data.get("observation_panel", "")) or str(data.get("pose_alignment", "")) != "observation_time":
        raise ValueError("Requires independent common-panel fields with observation-time pose")
    maps, occupancy = data[map_key], data["occupancy"]
    components = canonical_components(maps, occupancy)
    peaks = np.zeros_like(components)
    for target in range(maps.shape[-1]):
        if np.nanmax(maps[..., target]) > 0:
            index = np.unravel_index(np.nanargmax(maps[..., target]), maps.shape[:2])
            peaks[index[0], index[1], target] = True
    horizons = provenance.get("horizons", [8, 16, 32, 64])
    summary = {
        k: provenance[k]
        for k in ("schema", "workflow_version", "study_sha256", "checkpoint", "checkpoint_frames", "condition", "seed")
        if k in provenance
    }
    summary.update(
        fields_sha256=hashlib.sha256(fields.read_bytes()).hexdigest(),
        trials_sha256=hashlib.sha256(trials.read_bytes()).hexdigest(),
        observation_panel=str(data["observation_panel"]),
        destination_map_key=map_key,
        destination_definition="Selected map peak 100-unit bin and largest-mass half-peak component; fixed independently of intervention trajectories",
        efficiency_definition="Euclidean start-to-actual-arrival displacement / traveled path; not geodesic efficiency",
        limitation="Four starts and one seed per controller are diagnostic. Unrepresented identities remain missing, not successes or failures. Mono-component status alone does not establish a compact field.",
    )
    summary["command_effect_on_trajectory"] = trajectory_command_effect(rows)
    out.mkdir(parents=True, exist_ok=True)
    for name, masks in (("peak_bin", peaks), ("canonical_component", components)):
        records, contrasts, result = evaluate_destinations(rows, masks, horizons)
        pd.DataFrame(records).to_csv(out / (name + "_arrivals.csv"), index=False)
        pd.DataFrame(contrasts).to_csv(out / (name + "_paired_contrasts.csv"), index=False)
        summary[name] = result
    (out / "physical_destination_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trials", type=Path)
    parser.add_argument("fields", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--map-key", choices=("raw_dg_rate_maps", "rate_maps"), default="raw_dg_rate_maps")
    args = parser.parse_args()
    print(json.dumps(analyze(args.trials, args.fields, args.output, args.map_key), indent=2))


if __name__ == "__main__":
    main()
