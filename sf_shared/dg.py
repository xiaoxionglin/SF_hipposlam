"""Torch-only DG projection, extracted unchanged from IntrMotiv at 14ddead.

The experiment-local class remains available for opt-in rollout and rollback.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class DGProjection_batchnorm_relu(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        intercept=2,
        batchnorm_semantics: str = "legacy_batch",
        output_activation: str = "threshold_relu",
    ):
        """Normalize projected features, then optionally apply a sparse threshold."""
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.batchnorm1d = nn.BatchNorm1d(out_features, affine=False, momentum=0.1)
        self.activation = nn.ReLU()
        self.intercept = intercept
        if output_activation not in ("threshold_relu", "identity"):
            raise ValueError(f"Unknown DG output activation: {output_activation}")
        self.output_activation = output_activation
        if batchnorm_semantics not in (
            "legacy_batch",
            "running_consistent",
            "running_poststep_atomic",
            "input_centered_atomic",
        ):
            raise ValueError(f"Unknown DG BatchNorm semantics: {batchnorm_semantics}")
        self.batchnorm_semantics = batchnorm_semantics
        # This flag is controlled by the learner around its sole DG forward.
        # It deliberately defaults to false so actor inference, value bootstrap,
        # and decoder-only phases can never mutate the shared statistics.
        self._running_stats_update_enabled = False
        self.freeze_running_stats = False
        self.calibrate_frozen_on_first_batch = False
        self.last_running_stats_updated = False
        self.last_poststep_calibration_updated = False
        self.last_raw_logits: torch.Tensor | None = None
        self.last_pre_threshold_logits: torch.Tensor | None = None
        self._cached_projection_input: torch.Tensor | None = None
        self.register_buffer("feature_running_mean", torch.zeros(in_features))
        self.register_buffer("feature_num_batches_tracked", torch.zeros((), dtype=torch.int64))
        self.register_buffer("weight_generation", torch.zeros((), dtype=torch.int64))
        self.register_buffer("statistics_generation", torch.zeros((), dtype=torch.int64))
        # Learner-owned structural-plasticity state. These buffers are inert
        # unless orthogonal DG recruitment is explicitly enabled.
        # ``recruitment_committed`` is retained as legacy ever-recruited
        # telemetry. Graph recruitment uses graph evidence and birth support.
        self.register_buffer("recruitment_committed", torch.zeros(out_features, dtype=torch.bool))
        self.register_buffer("recruitment_activation_counts", torch.zeros(out_features))
        self.register_buffer("recruitment_row_counts", torch.zeros(out_features, dtype=torch.long))
        self.register_buffer("recruitment_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("recruitment_repeat_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("recruitment_tiny_residual_count", torch.zeros((), dtype=torch.long))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Checkpoints predating structural recruitment do not contain these
        # buffers. Supplying their initialized values keeps old runs loadable.
        row_counts_key = prefix + "recruitment_row_counts"
        if row_counts_key not in state_dict:
            committed = state_dict.get(prefix + "recruitment_committed", self.recruitment_committed)
            state_dict[row_counts_key] = committed.to(dtype=self.recruitment_row_counts.dtype).clone()
        for name in (
            "feature_running_mean",
            "feature_num_batches_tracked",
            "weight_generation",
            "statistics_generation",
            "recruitment_committed",
            "recruitment_activation_counts",
            "recruitment_count",
            "recruitment_repeat_count",
            "recruitment_tiny_residual_count",
        ):
            state_dict.setdefault(prefix + name, getattr(self, name))
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @contextmanager
    def running_stats_update(self, enabled: bool):
        previous = self._running_stats_update_enabled
        self._running_stats_update_enabled = bool(enabled)
        try:
            yield
        finally:
            self._running_stats_update_enabled = previous

    def _running_consistent_normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.last_running_stats_updated = False
        batchnorm = self.batchnorm1d
        if self.training and self._running_stats_update_enabled:
            with torch.no_grad():
                batch_mean = x.detach().mean(dim=0)
                batch_var = x.detach().var(dim=0, unbiased=False)
                if int(batchnorm.num_batches_tracked.item()) == 0:
                    batchnorm.running_mean.copy_(batch_mean)
                    batchnorm.running_var.copy_(batch_var)
                else:
                    momentum = float(batchnorm.momentum if batchnorm.momentum is not None else 0.1)
                    batchnorm.running_mean.lerp_(batch_mean, momentum)
                    batchnorm.running_var.lerp_(batch_var, momentum)
                batchnorm.num_batches_tracked.add_(1)
            self.last_running_stats_updated = True
        return F.batch_norm(
            x,
            batchnorm.running_mean,
            batchnorm.running_var,
            weight=None,
            bias=None,
            training=False,
            momentum=0.0,
            eps=batchnorm.eps,
        )

    @staticmethod
    @torch.no_grad()
    def _update_running_moments(batchnorm: nn.BatchNorm1d, values: torch.Tensor) -> None:
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        if int(batchnorm.num_batches_tracked.item()) == 0:
            batchnorm.running_mean.copy_(batch_mean)
            batchnorm.running_var.copy_(batch_var)
        else:
            momentum = float(batchnorm.momentum if batchnorm.momentum is not None else 0.1)
            batchnorm.running_mean.lerp_(batch_mean, momentum)
            batchnorm.running_var.lerp_(batch_var, momentum)
        batchnorm.num_batches_tracked.add_(1)

    def centered_projection_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.batchnorm_semantics == "input_centered_atomic":
            return x - self.feature_running_mean
        return x

    def project_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.centered_projection_input(x))

    @torch.no_grad()
    def post_step_update_running_stats(self, valids: torch.Tensor | None = None) -> bool:
        """Calibrate cheap DG moments after the updated weights are finalized.

        The learner caches detached DG inputs during its sole differentiable
        forward. This method performs no visual, CA3, or policy forward and is
        called under the same policy lock as the optimizer step.
        """
        self.last_poststep_calibration_updated = False
        if self.batchnorm_semantics not in ("running_poststep_atomic", "input_centered_atomic"):
            self._cached_projection_input = None
            return False
        inputs = self._cached_projection_input
        self._cached_projection_input = None
        if inputs is None:
            raise RuntimeError("Atomic DG calibration requires cached projection inputs")
        if valids is not None:
            valid = valids.bool().reshape(-1)
            if valid.numel() != inputs.size(0):
                raise ValueError("DG calibration valids must align with cached projection inputs")
            inputs = inputs[valid]
        if inputs.size(0) < 2:
            raise RuntimeError("Atomic DG calibration requires at least two valid inputs")

        if self.batchnorm_semantics == "input_centered_atomic":
            batch_mean = inputs.mean(dim=0)
            if int(self.feature_num_batches_tracked.item()) == 0:
                self.feature_running_mean.copy_(batch_mean)
            else:
                momentum = float(self.batchnorm1d.momentum if self.batchnorm1d.momentum is not None else 0.1)
                self.feature_running_mean.lerp_(batch_mean, momentum)
            self.feature_num_batches_tracked.add_(1)

        updated_logits = self.project_logits(inputs)
        self._update_running_moments(self.batchnorm1d, updated_logits)
        self.weight_generation.add_(1)
        self.statistics_generation.add_(1)
        self.last_poststep_calibration_updated = True
        return True

    @torch.no_grad()
    def mark_replacement_committed(self) -> None:
        """Record a row replacement as one atomic weight/statistics version."""
        self.weight_generation.add_(1)
        self.statistics_generation.add_(1)

    def preactivation(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            and self._running_stats_update_enabled
            and self.batchnorm_semantics in ("running_poststep_atomic", "input_centered_atomic")
        ):
            self._cached_projection_input = x.detach()
        x = self.project_logits(x)  # Shape: [batch_size, out_features]
        self.last_raw_logits = x
        if self.freeze_running_stats:
            self.last_running_stats_updated = False
            if (
                self.calibrate_frozen_on_first_batch
                and self.training
                and self._running_stats_update_enabled
                and int(self.batchnorm1d.num_batches_tracked.item()) == 0
            ):
                # Calibrate a fresh random projection from one unlabeled learner
                # minibatch, then keep both weights and moments fixed.
                self._update_running_moments(self.batchnorm1d, x.detach())
                self.last_running_stats_updated = True
            x = F.batch_norm(
                x,
                self.batchnorm1d.running_mean,
                self.batchnorm1d.running_var,
                weight=None,
                bias=None,
                training=False,
                momentum=0.0,
                eps=self.batchnorm1d.eps,
            )
        elif self.batchnorm_semantics == "running_consistent":
            x = self._running_consistent_normalize(x)
        elif self.batchnorm_semantics in ("running_poststep_atomic", "input_centered_atomic"):
            self.last_running_stats_updated = False
            x = F.batch_norm(
                x,
                self.batchnorm1d.running_mean,
                self.batchnorm1d.running_var,
                weight=None,
                bias=None,
                training=False,
                momentum=0.0,
                eps=self.batchnorm1d.eps,
            )
        else:
            self.last_running_stats_updated = bool(self.training)
            x = self.batchnorm1d(x)
        # The learner-only anti-Hebbian loss uses these logits. They stay on the
        # autograd graph and are overwritten by every encoder forward pass.
        self.last_pre_threshold_logits = x
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.preactivation(x)
        if self.output_activation == "identity":
            return normalized
        return self.activation(normalized - self.intercept)
