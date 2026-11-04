"""Held-out next-DG prediction on one immutable, shared trajectory dataset.

This is an offline representation diagnostic, not a policy improvement or a
counterfactual simulator. At horizon h it uses S_t and executed a_t..a_(t+h-1)
to predict u_(t+h). Entire episodes are held out; no target enters S_t.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sf_working_directories.IntrMotiv.dmlab.ca3_state_readout import (
    CA3StateReadout,
    CausalDGInnovationPredictor,
    _stratified_smooth_l1,
    _variance_covariance_losses,
)

SCHEMA = "intrmotiv/offline-ca3-prediction/v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(data, metadata):
    required = ("ca3", "actions", "dones_after", "episode_id", "step_in_episode")
    if any(key not in data for key in required):
        raise ValueError(f"Dataset requires {required}")
    count = len(data["ca3"])
    n, expanded, actions = (int(metadata[k]) for k in ("n_dg", "expanded", "action_count"))
    if min(count, n, expanded, actions) < 1 or data["ca3"].shape != (count, n * expanded):
        raise ValueError("Canonical CA3 shape does not match metadata")
    if any(data[k].shape != (count,) for k in required[1:]):
        raise ValueError("Trajectory fields must be one-dimensional and aligned")
    if data["dones_after"].dtype != np.bool_:
        raise ValueError("Reset flags must be boolean")
    if not np.isfinite(data["ca3"]).all() or (data["ca3"] < 0).any():
        raise ValueError("CA3 must contain finite nonnegative canonical activity")
    for key in ("actions", "episode_id", "step_in_episode"):
        if not np.issubdtype(data[key].dtype, np.integer):
            raise ValueError(f"{key} must be integral")
    if ((data["actions"] < 0) | (data["actions"] >= actions)).any():
        raise ValueError("Executed action outside declared action space")
    changed = np.r_[True, data["episode_id"][1:] != data["episode_id"][:-1]]
    if np.unique(data["episode_id"][changed]).size != changed.sum():
        raise ValueError("Each episode must be one contiguous trajectory")
    if not np.all(data["step_in_episode"][changed] == 0):
        raise ValueError("Each recorded episode must begin at its reset")
    continuation = ~changed[1:]
    if not np.all(np.diff(data["step_in_episode"])[continuation] == 1):
        raise ValueError("Missing or reordered decisions within an episode")
    if np.any(data["dones_after"][:-1][continuation]):
        raise ValueError("Episode contains observations after a reset")
    if not np.all(data["dones_after"][:-1][changed[1:]]):
        raise ValueError("Episode boundary lacks a done/censor marker")


def episode_splits(episode_ids, seed=901):
    """One fixed 60/20/20 split, shared by all horizons, models and fit seeds."""
    episodes = np.unique(episode_ids)
    if len(episodes) < 6:
        raise ValueError("At least six independently reset episodes are required")
    episodes = np.random.default_rng(seed).permutation(episodes)
    train_end, validation_end = int(len(episodes) * 0.6), int(len(episodes) * 0.8)
    return dict(train=episodes[:train_end], validation=episodes[train_end:validation_end], test=episodes[validation_end:])


def transition_indices(data, horizon):
    """Indices only: do not duplicate the potentially gigabyte-sized CA3 array."""
    if horizon < 1 or horizon >= len(data["ca3"]):
        raise ValueError("Horizon must be positive and shorter than the dataset")
    starts = np.arange(len(data["ca3"]) - horizon)
    valid = data["episode_id"][starts] == data["episode_id"][starts + horizon]
    for offset in range(horizon):
        valid &= ~data["dones_after"][starts + offset].astype(bool)
    return starts[valid]


def support(data, starts, expanded, action_count, horizon):
    target = data["ca3"][starts + horizon, ::expanded]
    return dict(
        transitions=len(starts), active_transitions=int((target > 0).any(1).sum()),
        active_elements=int((target > 0).sum()), zero_elements=int((target == 0).sum()),
        episodes=int(np.unique(data["episode_id"][starts]).size),
        first_action_counts=np.bincount(data["actions"][starts], minlength=action_count).tolist(),
        active_first_action_counts=np.bincount(data["actions"][starts][(target > 0).any(1)], minlength=action_count).tolist(),
        active_counts_by_dg=(target > 0).sum(0).tolist(),
    )


def support_gate(coverage, minimum_active=20):
    reasons = []
    for split, row in coverage.items():
        if row["active_transitions"] < minimum_active:
            reasons.append(f"{split}: active transitions {row['active_transitions']} < {minimum_active}")
        if not row["zero_elements"]:
            reasons.append(f"{split}: no zero-DG target elements")
        if sum(count > 0 for count in row["first_action_counts"]) < 2:
            reasons.append(f"{split}: fewer than two executed actions")
    return reasons


def prediction_metrics(prediction, target):
    """Stratification prevents the many inactive DG elements hiding errors."""
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite held-out prediction or target")
    active = target > 0
    square = (prediction - target).square()
    element = F.smooth_l1_loss(prediction, target, reduction="none")
    result = {}
    for label, mask in (("active", active), ("zero", ~active)):
        result[f"{label}_mse"] = float(square[mask].mean()) if mask.any() else None
        result[f"{label}_smooth_l1"] = float(element[mask].mean()) if mask.any() else None
    result["mse"] = float(square.mean())
    result["selection_loss"] = (
        result["active_smooth_l1"] + 0.1 * result["zero_smooth_l1"]
        if all(result[k] is not None for k in ("active_smooth_l1", "zero_smooth_l1")) else None
    )
    return result


def batch_tensors(states, actions, starts, horizon):
    rows = torch.as_tensor(starts, dtype=torch.long, device=states.device)
    offsets = torch.arange(horizon, device=states.device)
    return states[rows], actions[rows[:, None] + offsets], torch.full_like(rows, horizon)


@torch.no_grad()
def evaluate(readout, predictor, states, actions, starts, horizon, expanded, batch_size=1024):
    targets, predictions, shuffled, latents = [], [], [], []
    # Rotate complete action prefixes across held-out rows, never future states.
    shuffled_starts = np.roll(starts, max(1, len(starts) // 2))
    for offset in range(0, len(starts), batch_size):
        rows = starts[offset:offset + batch_size]
        state, prefix, h = batch_tensors(states, actions, rows, horizon)
        target = states[torch.as_tensor(rows + horizon, device=states.device), ::expanded]
        latent = None if readout is None else readout(state)
        pred = state[:, ::expanded] if readout is None else predictor(latent, prefix, h)
        targets.append(target.cpu())
        predictions.append(pred.cpu())
        if readout is not None:
            latents.append(latent.cpu())
            _, wrong, _ = batch_tensors(states, actions, shuffled_starts[offset:offset + batch_size], horizon)
            shuffled.append(predictor(readout(state), wrong, h).cpu())
    target, pred = torch.cat(targets), torch.cat(predictions)
    result = prediction_metrics(pred, target)
    if shuffled:
        altered = prediction_metrics(torch.cat(shuffled), target)
        result["action_shuffle_active_mse_delta"] = altered["active_mse"] - result["active_mse"]
        result["action_shuffle_mse_delta"] = altered["mse"] - result["mse"]
        standard_deviation = torch.cat(latents).std(0, unbiased=False)
        result["latent_std_mean"] = float(standard_deviation.mean())
        result["latent_std_min"] = float(standard_deviation.min())
    return result


def fit_pair(states, actions, indices, metadata, horizon, seed, epochs=20, batch_size=512, lr=0.001):
    """Equal initial tensors, optimizer settings and row order; action mask differs."""
    device = states.device
    torch.manual_seed(seed)
    readout = CA3StateReadout(states.shape[1], 16).to(device)
    predictor = CausalDGInnovationPredictor(16, metadata["action_count"], horizon, metadata["n_dg"], 128).to(device)
    models = {"action": (readout, predictor), "state_only": (copy.deepcopy(readout), copy.deepcopy(predictor))}
    models["state_only"][1].action_conditioned = False
    optimizers = {name: torch.optim.Adam([
        {"params": r.parameters(), "lr": lr * 0.2}, {"params": p.parameters(), "lr": lr},
    ]) for name, (r, p) in models.items()}
    best, records = {}, []
    rng = np.random.default_rng(seed + 71)
    expanded = metadata["expanded"]
    for epoch in range(epochs):
        order = rng.permutation(indices["train"])
        for offset in range(0, len(order), batch_size):
            rows = order[offset:offset + batch_size]
            state, prefix, h = batch_tensors(states, actions, rows, horizon)
            target = states[torch.as_tensor(rows + horizon, device=device), ::expanded]
            for name, (r, p) in models.items():
                latent = r(state.detach())
                prediction_loss, _, _ = _stratified_smooth_l1(p(latent, prefix, h), target.detach(), 1.0, 0.1)
                variance, covariance, _, _ = _variance_covariance_losses(latent)
                loss = prediction_loss + 0.1 * variance + 0.01 * covariance
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss: {name}, seed={seed}, horizon={horizon}")
                optimizers[name].zero_grad(set_to_none=True)
                loss.backward()
                optimizers[name].step()
        for name, (r, p) in models.items():
            metrics = evaluate(r, p, states, actions, indices["validation"], horizon, expanded)
            score = metrics["selection_loss"]
            if score is None or not np.isfinite(score):
                raise ValueError("Validation lacks finite supported selection loss")
            if name not in best or score < best[name][0]:
                best[name] = (score, epoch + 1, copy.deepcopy(r.state_dict()), copy.deepcopy(p.state_dict()))
        print(f"h={horizon} seed={seed} epoch={epoch + 1}/{epochs}", flush=True)
    for name, (r, p) in models.items():
        r.load_state_dict(best[name][2])
        p.load_state_dict(best[name][3])
        for split in (key for key in indices if key != "train"):
            metrics = evaluate(r, p, states, actions, indices[split], horizon, expanded)
            records.append(dict(horizon=horizon, seed=seed, model=name, split=split, best_epoch=best[name][1], **metrics))
    return records, {name: dict(readout=value[2], predictor=value[3], epoch=value[1]) for name, value in best.items()}


def run_fit(args):
    metadata = json.loads(args.dataset.with_suffix(".json").read_text())
    if metadata["dataset_sha256"] != file_sha256(args.dataset):
        raise ValueError("Dataset hash differs from its collection provenance")
    with np.load(args.dataset, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    validate_dataset(data, metadata)
    splits = episode_splits(data["episode_id"], args.split_seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema=SCHEMA, dataset=str(args.dataset.resolve()), dataset_sha256=metadata["dataset_sha256"],
                  analysis_code_sha256=file_sha256(__file__),
                  collection=metadata, split_seed=args.split_seed, episodes={k: v.tolist() for k, v in splits.items()},
                  fit_settings=dict(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seeds=args.seeds,
                                    state_dim=16, hidden_size=128, readout_lr_scale=0.2,
                                    active_coeff=1.0, zero_coeff=0.1, var_coeff=0.1, cov_coeff=0.01),
                  coverage={}, unsupported={}, subgroup_unsupported={}, records=[], passed=False,
                  pass_meaning="All requested horizons have sufficient support and finite fits; no predictive advantage is implied.",
                  claim_limit="Executed-action held-out prediction only; no policy update, planning, or counterfactual identification.")
    states = torch.from_numpy(data["ca3"]).to(args.device)
    actions = torch.from_numpy(data["actions"]).long().to(args.device)
    for horizon in args.horizons:
        valid = transition_indices(data, horizon)
        indices = {split: valid[np.isin(data["episode_id"][valid], episodes)] for split, episodes in splits.items()}
        coverage = {split: support(data, rows, metadata["expanded"], metadata["action_count"], horizon) for split, rows in indices.items()}
        report["coverage"][str(horizon)] = coverage
        reasons = support_gate(coverage, args.minimum_active)
        if reasons:
            report["unsupported"][str(horizon)] = reasons
            continue
        # At H1 only, randomized a_t is independent of the actor's visual input.
        # For H16 later actions depend on later observations, so this first-action
        # subset would not make the whole recorded prefix an intervention.
        if horizon == 1 and "random_action" in data:
            rows = indices["test"][data["random_action"][indices["test"]].astype(bool)]
            subgroup = support(data, rows, metadata["expanded"], metadata["action_count"], horizon)
            report["coverage"][str(horizon)]["test_random_action"] = subgroup
            missing = support_gate({"test_random_action": subgroup}, args.minimum_active)
            if missing:
                report["subgroup_unsupported"]["test_random_action"] = missing
            else:
                indices["test_random_action"] = rows
        for split in (key for key in indices if key != "train"):
            metrics = evaluate(None, None, states, actions, indices[split], horizon, metadata["expanded"])
            report["records"].append(dict(horizon=horizon, seed=None, model="persistence", split=split, **metrics))
        for seed in args.seeds:
            rows, fitted = fit_pair(states, actions, indices, metadata, horizon, seed, args.epochs, args.batch_size, args.lr)
            report["records"].extend(rows)
            torch.save(dict(schema=SCHEMA, dataset_sha256=metadata["dataset_sha256"], horizon=horizon, seed=seed,
                            models=fitted, metadata=metadata), output / f"predictors_h{horizon}_s{seed}.pth")
        (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    report["passed"] = not report["unsupported"]
    report["test_comparisons"] = []
    for horizon in args.horizons:
        rows = [row for row in report["records"] if row["horizon"] == horizon and row["split"] == "test"]
        if not rows:
            continue
        by_model = {name: [row for row in rows if row["model"] == name] for name in ("action", "state_only", "persistence")}
        means = {name: float(np.mean([row["active_mse"] for row in group])) for name, group in by_model.items()}
        report["test_comparisons"].append(dict(horizon=horizon, mean_active_mse=means,
            action_minus_state_only=means["action"] - means["state_only"],
            action_minus_persistence=means["action"] - means["persistence"]))
    (output / "results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 16])
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 29, 47])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--split-seed", type=int, default=901)
    parser.add_argument("--minimum-active", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if 1 not in args.horizons or min(args.epochs, args.batch_size, args.minimum_active, args.lr) <= 0:
        parser.error("H1 is required and training/support parameters must be positive")
    return run_fit(args)


if __name__ == "__main__":
    raise SystemExit(main())
