"""Source-normalized, frozen features preserving the original depth bypass."""

import torch

from .contracts import canonical_events
from .replay import Observation


class FrozenParentFeatures:
    allowed = frozenset(("obs", "INSTR"))

    def __init__(self, actor, exclusive):
        self.actor = actor.eval().requires_grad_(False)
        self.exclusive = exclusive
        self.parity_verified = False
        self.n = actor.core.Hippo_n_feature
        encoder = actor.encoder
        if not encoder.depth_sensor or not encoder.bypass or encoder.goal_reference_projection is not None:
            raise ValueError("unsupported parent feature contract")
        if encoder.context_feedback != "none" or encoder.action_path_integration or encoder.context_action_count:
            raise ValueError("unsupported recurrent encoder history")

    def __call__(self, observation):
        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
        from sf_working_directories.IntrMotiv.dmlab.dmlab30 import DMLAB_INSTRUCTIONS

        allowed = ("obs", DMLAB_INSTRUCTIONS)
        obs = {key: observation[key] for key in allowed}
        self.actor.eval()
        with torch.no_grad():
            obs = prepare_and_normalize_obs(self.actor, obs)
            encoder = self.actor.encoder
            feature = encoder.projection_input(obs)
            pre = encoder.DG_projection.preactivation(feature)
            activity = encoder.DG_projection.activation(pre - encoder.DG_projection.intercept)
            depth = encoder.depth_encoder(obs["obs"][:, -1:]).flatten(1)
            bypass = torch.cat((depth, feature[:, -encoder.instructions_lstm_units :]), -1)
            events = canonical_events(activity, self.exclusive)
            if not self.parity_verified:
                reconstructed = torch.cat((activity, bypass), -1)
                if encoder.dg_goal_write:
                    reconstructed = torch.cat((reconstructed, pre), -1)
                source = self.actor.forward_head(obs)
                if not torch.equal(source, reconstructed):
                    raise RuntimeError("source head/feature adapter parity failure")
                self.parity_verified = True
        return [Observation(p, d, e) for p, d, e in zip(pre, bypass, events)]
