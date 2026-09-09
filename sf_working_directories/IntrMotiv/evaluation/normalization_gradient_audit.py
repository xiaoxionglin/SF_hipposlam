"""Audit DG normalization gradients on frozen visual features from one real rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from sf_working_directories.IntrMotiv.dmlab.custom_encoder import DGProjection_batchnorm_relu
from sf_working_directories.IntrMotiv.evaluation.place_fields import rollout_dg


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.reshape(-1)
    right = right.reshape(-1)
    denominator = left.norm() * right.norm()
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.dot(left, right) / denominator)


def _row_cosine_mean(rows: torch.Tensor) -> float:
    normalized = F.normalize(rows, dim=1)
    cosine = normalized @ normalized.T
    off_diagonal = ~torch.eye(rows.size(0), dtype=torch.bool)
    return float(cosine[off_diagonal].mean())


def _normalized_logits(
    features: torch.Tensor,
    weight: torch.Tensor,
    mode: str,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    if mode == "input_centered_atomic":
        features = features - features.mean(dim=0, keepdim=True).detach()
    raw = F.linear(features, weight)
    if mode == "legacy_batch":
        mean = raw.mean(dim=0)
        var = raw.var(dim=0, unbiased=False)
    elif mode in ("running_poststep_atomic", "input_centered_atomic"):
        mean = raw.mean(dim=0).detach()
        var = raw.var(dim=0, unbiased=False).detach()
    elif mode == "running_current":
        mean = running_mean
        var = running_var
    else:
        raise ValueError(mode)
    return (raw - mean) / torch.sqrt(var + eps)


def audit(
    features: torch.Tensor,
    projection: DGProjection_batchnorm_relu,
    learning_rate: float,
) -> dict:
    features = features.float()
    initial_weight = projection.linear.weight.detach().float()
    running_mean = projection.batchnorm1d.running_mean.detach().float()
    running_var = projection.batchnorm1d.running_var.detach().float()
    eps = float(projection.batchnorm1d.eps)
    intercept = float(projection.intercept)

    with torch.no_grad():
        legacy_reference = _normalized_logits(
            features,
            initial_weight,
            "legacy_batch",
            running_mean,
            running_var,
            eps,
        )
        credit_mask = legacy_reference > intercept
        if int(credit_mask.sum()) == 0:
            raise RuntimeError("Reference rollout produced no active DG events")
        reward = 0.1 * (1 + torch.arange(features.size(0), dtype=features.dtype) % 64)
        reward = reward.unsqueeze(1)

    results = {}
    gradients = {}
    logits_by_mode = {}
    for mode in (
        "legacy_batch",
        "running_current",
        "running_poststep_atomic",
        "input_centered_atomic",
    ):
        weight = initial_weight.clone().requires_grad_(True)
        logits = _normalized_logits(features, weight, mode, running_mean, running_var, eps)
        loss = -(logits * credit_mask * reward).sum() / credit_mask.sum()
        gradient = torch.autograd.grad(loss, weight)[0].detach()
        gradients[mode] = gradient
        logits_by_mode[mode] = logits.detach()

        updated_weight = F.normalize(initial_weight - float(learning_rate) * gradient, dim=1)
        with torch.no_grad():
            if mode == "running_current":
                updated_logits = _normalized_logits(
                    features, updated_weight, mode, running_mean, running_var, eps
                )
            else:
                updated_logits = _normalized_logits(
                    features, updated_weight, mode, running_mean, running_var, eps
                )
            updated_active = updated_logits > intercept
            row_mean = gradient.mean(dim=0)
            feature_mean = features.mean(dim=0)
            row_to_feature_mean = [
                _cosine(row, feature_mean) for row in gradient
            ]
        results[mode] = {
            "loss": float(loss.detach()),
            "active_fraction": float((logits.detach() > intercept).float().mean()),
            "active_mask_agreement_with_legacy": float(
                ((logits.detach() > intercept) == credit_mask).float().mean()
            ),
            "gradient_norm": float(gradient.norm()),
            "gradient_row_pair_cosine_mean": _row_cosine_mean(gradient),
            "gradient_mean_row_to_feature_mean_cosine": float(
                torch.tensor(row_to_feature_mean).mean()
            ),
            "gradient_abs_row_to_feature_mean_cosine": float(
                torch.tensor(row_to_feature_mean).abs().mean()
            ),
            "gradient_common_direction_norm": float(row_mean.norm()),
            "one_step_active_mask_change_fraction": float(
                (updated_active != (logits.detach() > intercept)).float().mean()
            ),
        }

    legacy_logits = logits_by_mode["legacy_batch"]
    legacy_gradient = gradients["legacy_batch"]
    for mode, values in results.items():
        values["forward_max_abs_difference_from_legacy"] = float(
            (logits_by_mode[mode] - legacy_logits).abs().max()
        )
        values["gradient_cosine_with_legacy"] = _cosine(gradients[mode], legacy_gradient)

    return {
        "feature_count": int(features.size(0)),
        "feature_dimension": int(features.size(1)),
        "feature_mean_norm": float(features.mean(dim=0).norm()),
        "feature_centered_rms": float(
            (features - features.mean(dim=0, keepdim=True)).square().mean().sqrt()
        ),
        "reference_credit_event_count": int(credit_mask.sum()),
        "intercept": intercept,
        "learning_rate": float(learning_rate),
        "modes": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-num-frames", type=int, default=32768)
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--checkpoint-rank", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    args = parser.parse_args()

    captured_features = []
    captured_projection = []
    original_forward = DGProjection_batchnorm_relu.forward

    def capture_forward(self, x):
        if len(captured_features) < int(args.max_features):
            remaining = int(args.max_features) - len(captured_features)
            captured_features.extend(x.detach().cpu()[:remaining])
            captured_projection[:] = [self]
        return original_forward(self, x)

    DGProjection_batchnorm_relu.forward = capture_forward
    try:
        _, checkpoint, *_ = rollout_dg(
            args.run_dir,
            max_num_frames=int(args.max_num_frames),
            deterministic=False,
            checkpoint_rank=int(args.checkpoint_rank),
        )
    finally:
        DGProjection_batchnorm_relu.forward = original_forward

    if not captured_features or not captured_projection:
        raise RuntimeError("No DG projection inputs were captured")
    features = torch.stack(captured_features[: int(args.max_features)])
    result = audit(features, captured_projection[0], float(args.learning_rate))
    result["run_dir"] = str(args.run_dir)
    result["checkpoint"] = str(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
