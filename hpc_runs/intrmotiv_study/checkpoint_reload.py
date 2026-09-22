"""Certify a real learner reload from an immutable checkpoint copy.

Run on a compute node with the training runtime on PYTHONPATH. This generalizes
the historical controller_compatibility_exact_restore.py qualification driver.
No learning or checkpoint writes occur in the original experiment directory.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch


def assert_exact(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left.to(right.device), right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right, (left, right)


def certify(run_dir, checkpoint, output):
    from sample_factory.utils.attr_dict import AttrDict
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.model_sharing import ParameterServer
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components
    from sf_working_directories.IntrMotiv.dmlab.controller_learner import ControllerLearner
    from sf_working_directories.IntrMotiv.dmlab.custom_learner import DistanceLearnerReward
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict

    output.mkdir(parents=True, exist_ok=False)
    immutable = output / checkpoint.name
    shutil.copy2(checkpoint, immutable)
    cfg = AttrDict(json.loads((run_dir / "config.json").read_text()))
    cfg.train_dir = str(output / "learner")
    cfg.cli_args = {}
    cfg.with_wandb = False
    # The temporary qualification environment must not consume training seeds.
    cfg.dmlab_use_level_cache = False
    target = Path(cfg.train_dir) / cfg.experiment / "checkpoint_p0"
    target.mkdir(parents=True)
    os.link(immutable, target / checkpoint.name)
    register_dmlab_components()
    torch.set_num_threads(1)
    env = make_env_func_batched(cfg, env_config=None)
    try:
        info = extract_env_info(env, cfg)
    finally:
        env.close()
    versions = torch.zeros(1, dtype=torch.int32)
    server = ParameterServer(0, versions, False)
    mode = getattr(cfg, "controller_learning", "ppo")
    cls = ControllerLearner if mode == "ddqn" else DistanceLearnerReward
    learner = cls(cfg, info, versions, 0, server)
    learner.init()
    saved = load_checkpoint_dict(immutable, torch.device("cpu"))
    assert_exact(saved["model"], learner.actor_critic.state_dict())
    assert_exact(saved["optimizer"], learner.optimizer.state_dict())
    assert learner.env_steps == saved["env_steps"]
    assert learner.train_step == saved["train_step"]
    if "controller" in saved:
        controller = saved["controller"]
        assert_exact(controller["target"], learner.target_snapshot.model.state_dict())
        assert_exact(controller["clock"], learner.clock.state_dict())
        assert_exact(controller["replay"]["rng"], learner.replay.rng.bit_generator.state)
        assert_exact(controller["her_rng"], learner.her_rng.bit_generator.state)
        assert_exact(controller["torch_rng"], torch.get_rng_state())
        assert_exact(controller["python_rng"], random.getstate())
        numpy_rng = controller["numpy_rng"]
        assert_exact((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]), np.random.get_state())
        assert learner.replay.session == controller["replay"]["session"] + 1
        assert learner.publication == controller["publication"]
        assert learner.controller_version == controller["version"]
        assert learner.target_snapshot.version == controller["target_version"]
        assert learner.fresh_dg_steps == controller["fresh_dg_steps"]
        assert learner.fresh_graph_batches == controller["fresh_graph_batches"]
        assert learner.replay.ingress.pending == 0
        assert learner.replay.rejected.get("restart_pending_tail", 0) == (
            controller["replay"]["rejected"].get("restart_pending_tail", 0) + controller["replay"]["pending"]
        )
        assert learner.frame_milestones == set(controller.get("frame_milestones", ()))
        restored_replay = learner.replay.state_dict()
        for key in ("rows", "capacity", "received", "accepted", "physical_frames"):
            assert_exact(controller["replay"][key], restored_replay[key])
        assert_exact(saved["model"], learner.published.state_dict())
    digest = hashlib.sha256()
    with immutable.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    result = dict(
        run=cfg.experiment, checkpoint=str(immutable), checkpoint_sha256=digest.hexdigest(),
        frames=learner.env_steps, train_step=learner.train_step, controller=mode,
        exact_restore=True, model_and_buffers_exact=True, optimizer_exact=True,
        counters_exact=True, device=str(learner.device), job_id=os.environ.get("SLURM_JOB_ID"),
    )
    (output / "certificate.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    certify(args.run_dir.resolve(), args.checkpoint.resolve(), args.output_dir.resolve())


if __name__ == "__main__":
    main()
