"""Strict model conversion after the source actor has loaded its full state."""

import hashlib
import json
from pathlib import Path

import torch

from .worker import QWorker


def state_hash(module):
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(str((str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_parent(cfg, action_vectors):
    required = dict(
        encoder_conv_architecture="layer2_resnet18",
        Hippo_R=8,
        Hippo_L=64,
        DG_name="batchnorm_relu",
        hrl_goal_conditioning="target_id_film",
        env_frameskip=4,
        depth_sensor=True,
        with_number_instruction=True,
        hrl_target_timing="immediate",
        dg_context_feedback="none",
    )
    for key, expected in required.items():
        if getattr(cfg, key, None) != expected:
            raise ValueError(f"unsupported parent {key}: {getattr(cfg,key,None)!r}; expected {expected!r}")
    if int(cfg.Hippo_n_feature) not in (16, 64) or getattr(cfg, "dg_goal_input", "none") not in ("none", "write"):
        raise ValueError("unsupported DG configuration")
    for key in (
        "hrl_action_path_integration",
        "hrl_motion_policy_input",
        "hrl_behavior_mode_condition",
        "dg_orthogonal_recruitment",
    ):
        if getattr(cfg, key, False):
            raise ValueError(f"unsupported history or decoder input: {key}")
    expected = [
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, -1, 0, 0, 0],
        [0, 0, -1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [-20, 0, 0, 0, 0, 0, 0],
        [20, 0, 0, 0, 0, 0, 0],
        [-20, 0, 0, 1, 0, 0, 0],
        [20, 0, 0, 1, 0, 0, 0],
    ]
    if torch.as_tensor(action_vectors).tolist() != expected:
        raise ValueError("ordered action vectors differ from the parent study")


def convert_actor(actor, cfg, action_vectors, learner_seed):
    validate_parent(cfg, action_vectors)
    core = actor.core
    n = int(cfg.Hippo_n_feature)
    if core.get_out_size() != core.core_output_size + core.bypass_size + n:
        raise ValueError("unaccounted parent decoder inputs")
    actor.eval().requires_grad_(False)
    with torch.random.fork_rng():
        torch.manual_seed(learner_seed)
        worker = QWorker(
            actor.decoder,
            n,
            core.bypass_size,
            core.R,
            int(cfg.Hippo_L),
            getattr(core, "dg_goal_modulation", None),
            float(cfg.DG_BN_intercept),
            batch_independent=type(actor.decoder).__name__ == "TargetFiLMDecoder",
        )
    worker.requires_grad_(True)
    copied = [k for k in actor.state_dict() if k.startswith(("encoder.", "obs_normalizer.", "decoder."))]
    copied += [k for k in actor.state_dict() if k == "core.dg_goal_modulation"]
    report = dict(
        schema="intrmotiv/ddqn-conversion/v2",
        learner_seed=learner_seed,
        copied=copied,
        excluded=[k for k in actor.state_dict() if k not in copied],
        initialized=["q_head.weight", "q_head.bias", "joint.0.weight", "joint.0.bias"],
        encoder_hash=state_hash(actor.encoder),
        worker_hash=state_hash(worker),
        parent_frames_are_child_frames=False,
        graph_learning=False,
        dg_update_mode="frozen",
        decoder_parameter_count=sum(p.numel() for p in worker.parameters()),
        recognition="exclusive" if core.topological_enabled else "dominant_positive",
        budget_reference=64,
        prefix_width=core.expanded_length,
    )
    return worker, report
