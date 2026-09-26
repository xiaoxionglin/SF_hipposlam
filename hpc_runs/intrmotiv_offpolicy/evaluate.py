"""Adapt a frozen DDQN readout to the canonical exact-start command evaluator."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoint import convert_actor, state_hash


def validate_child_schema(child):
    if child.get("schema") != "intrmotiv/ddqn-worker/v2":
        raise ValueError("incompatible child checkpoint: v2 requires fresh initialization; no v1 migration")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parent-run-dir", type=Path, required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--child-checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--decision-cap", type=int, default=100000)
    p.add_argument("--registry", default="1,4,11")
    p.add_argument("--max-sources", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    output = args.output.resolve()
    output.relative_to(Path("/work/classic/fr_xl1014-train"))
    output.mkdir(parents=True, exist_ok=False)
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict, load_policy_env
    from sf_working_directories.IntrMotiv.evaluation.target_control_interventions import (
        run_landmark_matched_interventions,
    )

    cfg, env, env_info, actor, path, device = load_policy_env(
        args.parent_run_dir, args.decision_cap * 4, True, 0, args.parent_checkpoint
    )
    callback = None
    q_statistics = []
    if args.child_checkpoint:
        with torch.serialization.safe_globals([type(Path("."))]):
            child = load_checkpoint_dict(args.child_checkpoint, device)
        validate_child_schema(child)
        if cfg.dg_goal_input != "none":
            raise ValueError("write-conditioned actor evaluation not qualified")
        worker, _ = convert_actor(actor, cfg, env.unwrapped.action_list, int(child["config"]["seed"]))
        worker.load_state_dict(child["model"], strict=True)
        worker.eval().requires_grad_(False)
        if state_hash(actor.encoder) != child["reference_hash"]:
            raise ValueError("reference detector changed")

        def child_policy_step(out, command, budget, clock):
            state = out[:, : actor.core.get_out_size()]
            q = worker.readout_state(state, out.new_tensor([budget]), out.new_tensor([clock]))
            q_statistics.append(
                (float(q.min()), float(q.max()), float(q.mean()), int(((q < 0) | (q > 1)).sum()), q.numel())
            )
            logits = torch.full_like(q, -torch.inf).scatter(1, q.argmax(-1, keepdim=True), 0.0)
            return {"action_logits": logits}

        callback = child_policy_step

    rows, summary = run_landmark_matched_interventions(
        cfg,
        env,
        env_info,
        actor,
        path,
        device,
        args.decision_cap,
        deterministic=True,
        max_sources=args.max_sources,
        targets_per_source=3,
        repeats=args.repeats,
        policy_step=callback,
        deadline_override=64,
        target_registry=[int(x) for x in args.registry.split(",")],
    )
    rows.to_csv(output / "trials.csv", index=False)
    if len(rows):
        commanded = rows[rows.commanded]
        summary.update(
            commanded_successes=int(commanded.hit.sum()),
            commanded_trials=len(commanded),
            commanded_arrival_lower_bound=float(commanded.hit.mean()),
            censored_commanded_trials=int(commanded.censored.sum()),
            restricted_mean_first_arrival=float(
                commanded.apply(lambda row: row.hit_time if row.hit else row.deadline, axis=1).mean()
            ),
        )
    if len(rows):
        summary["per_goal"] = {
            str(goal): dict(
                successes=int(group.hit.sum()),
                trials=len(group),
                censored=int(group.censored.sum()),
                success_lower_bound=float(group.hit.mean()),
                failure_inclusive_arrival_time=float(
                    group.apply(lambda row: row.hit_time if row.hit else row.deadline, axis=1).mean()
                ),
            )
            for goal, group in rows[rows.commanded].groupby("target")
        }
    summary["child_seed"] = int(child["config"]["seed"]) if args.child_checkpoint else None
    summary.update(
        child_checkpoint=str(args.child_checkpoint) if args.child_checkpoint else None,
        independent_spatial_destination_qualification=False,
        registry_scope="post_hoc_development",
        physical_prefixes="same seeds and passive source inventory across parent/child",
    )
    if q_statistics:
        summary.update(
            q_min=min(s[0] for s in q_statistics),
            q_max=max(s[1] for s in q_statistics),
            q_mean=sum(s[2] for s in q_statistics) / len(q_statistics),
            q_out_of_range_fraction=sum(s[3] for s in q_statistics) / sum(s[4] for s in q_statistics),
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
