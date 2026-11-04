"""Exercise frozen episode and matched-prefix evaluation on a real checkpoint.

The prefix check qualifies execution independently of whether learned landmarks
provide eligible scientific trials. It does not report a control-performance
score. Scientific matched-landmark interventions remain the canonical evaluator.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .checkpoint_reload import assert_exact
from .geometry import geometry_from_config


def qualify(run_dir, checkpoint, output):
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs, make_dones
    from sample_factory.algo.sampling.batched_sampling import preprocess_actions
    from sample_factory.model.model_utils import get_rnn_size
    from sample_factory.utils.attr_dict import AttrDict
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_policy_env, evaluation_pose
    from sf_working_directories.IntrMotiv.evaluation.episode_coverage import evaluate_episodes
    from sf_working_directories.IntrMotiv.evaluation.target_control_interventions import condition_for_target

    torch.set_num_threads(1)
    cfg, env, info, actor, _, device = load_policy_env(run_dir, 10000, False, 0, checkpoint)
    env.close()
    cfg.dmlab_use_level_cache = False
    geometry = geometry_from_config(cfg)
    assert geometry is not None
    assert cfg.with_pos_obs is False
    before = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    actions = np.random.default_rng(51000).integers(0, actor.action_space.n, 64)
    signatures = []
    with torch.no_grad():
        for repeat in range(2):
            env = make_env_func_batched(
                cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=None
            )
            try:
                env.unwrapped.reset_on_init = False
                env.unwrapped.seed(51000)
                torch.manual_seed(61000)
                obs, _ = env.reset()
                state = torch.zeros((1, get_rnn_size(cfg)), device=device)
                for action in actions:
                    assert not ({"pos", "rot", "GEOMETRY.ENTITY_LAYER"} & obs.keys())
                    head = actor.forward_head(prepare_and_normalize_obs(actor, obs))
                    _, state = actor.forward_core(head, state)
                    obs, reward, term, trunc, _ = env.step(
                        preprocess_actions(info, torch.tensor([[int(action)]], device=device))
                    )
                    assert float(reward[0]) == 0 and not bool(make_dones(term, trunc)[0])
                head = actor.forward_head(prepare_and_normalize_obs(actor, obs))
                core, next_state = actor.forward_core(head, state)
                command_logits = []
                for command in range(min(4, int(cfg.Hippo_n_feature))):
                    conditioned = condition_for_target(actor, core, -1, command)
                    result = actor.forward_tail(conditioned, values_only=False, sample_actions=False)
                    assert torch.isfinite(result["action_logits"]).all()
                    command_logits.append(result["action_logits"].clone())
                signatures.append({
                    "observations": {k: v.clone() for k, v in obs.items()},
                    "pose": tuple(v.clone() for v in evaluation_pose(env, obs)),
                    "state": next_state.clone(), "command_logits": command_logits,
                })
                assert env.unwrapped.geometry_verified
            finally:
                env.close()
        assert_exact(signatures[0], signatures[1])
        assert_exact(before, actor.state_dict())
    policy = evaluate_episodes(cfg, actor, info, output, episodes=2)
    random = evaluate_episodes(cfg, actor, info, output, episodes=2, random_actions=True)
    for probe in (policy, random):
        assert all(1799 <= row["decisions"] <= 1801 for row in probe["episodes"])
        assert all(row["geometry_invalid_pose_steps"] == 0 for row in probe["episodes"])
    assert_exact(before, actor.state_dict())
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = dict(
        checkpoint=str(checkpoint), checkpoint_sha256=digest.hexdigest(),
        geometry_sha256=geometry["sha256"], exact_prefix_verified=True,
        policy_and_graph_frozen=True, privileged_inputs_absent=True,
        matched_reset_seeds=[51000, 51001], complete_policy_episodes=2,
        complete_random_episodes=2, passed=True,
    )
    (output / "evaluation_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.run_dir, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
