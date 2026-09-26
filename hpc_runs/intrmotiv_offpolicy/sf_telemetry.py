"""Connect native DDQN behavior to existing IntrMotiv telemetry and SF reports."""

import numbers
from collections import defaultdict

import numpy as np
import torch


class NativeTelemetry:
    def __init__(self, cfg, env_info, policy_id, actor):
        from sf_working_directories.IntrMotiv.dmlab.online_spatial_telemetry import TrainingSpatialTelemetry

        # No active graph learner: do not report inherited parent graph buffers.
        self.spatial = TrainingSpatialTelemetry(cfg, env_info, policy_id, 0, None)
        self.projection = actor.parent.encoder.DG_projection
        self.n = actor.worker.n_goals

    def capture(self, batch, policy_id, frames):
        from sf_working_directories.IntrMotiv.dmlab.custom_learner import dg_usage_metrics

        # Cached frozen preactivations reconstruct the exact behavior activity.
        # Never use exclusive recognition events as multi-unit DG activity.
        with torch.no_grad():
            activity = self.projection.activation(batch["ddqn_packet"][..., : self.n] - self.projection.intercept)
        valids = batch["policy_id"] == policy_id
        buff = dict(batch, dg_activity=activity)
        self.spatial.append_batch(buff, valids)
        stats = self.spatial.on_env_steps(frames)
        # The shared coordinator's graph defaults are zeros; absent graph is N/A.
        stats = {k: v for k, v in stats.items() if not k.startswith("online_spatial_graph_")}
        active = activity[valids] > 0
        if active.numel():
            low, mean, high, entropy = dg_usage_metrics(active)
            stats.update(
                dg_density=float(active.float().mean()),
                dg_multi_activation_rate=float((active.sum(-1) > 1).float().mean()),
                dg_silent_unit_frac=float((~active.any(0)).float().mean()),
                dg_unit_duty_cycle_min=float(low),
                dg_unit_duty_cycle_mean=float(mean),
                dg_unit_duty_cycle_max=float(high),
                dg_usage_entropy=float(entropy),
            )
        return stats


def extra_reports(infos, dones, policy_id):
    """Keep existing environment measurements; SF's list-info branch skips them."""
    from sample_factory.algo.utils.misc import EPISODIC, POLICY_ID_KEY

    values = defaultdict(list)
    if not isinstance(infos, (list, tuple)):
        return []
    for info, done in zip(infos, dones):
        for kind in ("periodic_stats", "episode_extra_stats"):
            if kind == "episode_extra_stats" and not bool(done):
                continue
            for key, value in info.get(kind, {}).items():
                if isinstance(value, numbers.Number) and np.isfinite(value):
                    values[key].append(float(value))
    return [{EPISODIC: {k: np.asarray(v) for k, v in values.items()}, POLICY_ID_KEY: policy_id}] if values else []


def install_sampling_reports():
    from sample_factory.algo.sampling.batched_sampling import BatchedVectorEnvRunner

    original = BatchedVectorEnvRunner._process_env_step
    if getattr(original, "_ddqn_telemetry", False):
        return

    def process(self, rewards, dones, infos):
        reports = original(self, rewards, dones, infos)
        if getattr(self.cfg, "ddqn_telemetry", False):
            reports.extend(extra_reports(infos, dones.cpu().tolist(), self.policy_id))
        return reports

    process._ddqn_telemetry = True
    BatchedVectorEnvRunner._process_env_step = process
