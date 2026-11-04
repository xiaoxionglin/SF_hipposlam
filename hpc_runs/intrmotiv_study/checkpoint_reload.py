"""Certify a real learner reload from an immutable checkpoint copy.

Run on a compute node with the training runtime on PYTHONPATH. This generalizes
the historical controller_compatibility_exact_restore.py qualification driver.
No learning or checkpoint writes occur in the original experiment directory.
"""

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

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


def assert_optimizer_restore(saved, restored, saved_curr_lr, restored_curr_lr):
    """Verify exact state plus Learner.init's explicit scheduled-LR reset.

    A checkpoint can retain a last-minibatch LR reduced for invalid samples.
    Learner.init loads that state, then calls _apply_lr(curr_lr). Reproduce only
    that known transition; every other optimizer field remains bitwise exact.
    Raw equality is reported separately and is never claimed after an LR reset.
    """
    assert_exact(saved_curr_lr, restored_curr_lr)
    expected = copy.deepcopy(saved)
    groups = expected["param_groups"]
    assert groups, "optimizer has no parameter groups"
    first_scale = float(groups[0].get("intrmotiv_lr_scale", 1.0))
    assert first_scale > 0 and np.isfinite(first_scale)
    assert np.isfinite(saved_curr_lr) and saved_curr_lr >= 0
    reset = saved_curr_lr != groups[0]["lr"] / first_scale
    if reset:
        for group in groups:
            scale = float(group.get("intrmotiv_lr_scale", 1.0))
            assert scale > 0 and np.isfinite(scale)
            group["lr"] = saved_curr_lr * scale
    assert_exact(expected, restored)
    assert_exact(saved["state"], restored["state"])
    before_lrs = [group["lr"] for group in saved["param_groups"]]
    after_lrs = [group["lr"] for group in restored["param_groups"]]
    return dict(
        optimizer_exact=before_lrs == after_lrs,
        optimizer_state_exact=True,
        optimizer_runtime_contract_exact=True,
        optimizer_lr_reset_verified=True,
        optimizer_lr_reset_applied=reset,
        optimizer_lr_before_init=before_lrs,
        optimizer_lr_after_init=after_lrs,
        optimizer_scheduled_lr=saved_curr_lr,
    )


def certify(run_dir, checkpoint, output):
    from sample_factory.algo.utils.env_info import extract_env_info
    from sample_factory.algo.utils.make_env import make_env_func_batched
    from sample_factory.algo.utils.model_sharing import ParameterServer
    from sample_factory.utils.attr_dict import AttrDict
    from sf_working_directories.IntrMotiv.dmlab.custom_learner import make_hipposlam_learner
    from sf_working_directories.IntrMotiv.dmlab.train_hipposlam import register_dmlab_components
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict

    started = time.monotonic()
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
    environment_started = time.monotonic()
    env = make_env_func_batched(cfg, env_config=None)
    try:
        info = extract_env_info(env, cfg)
    finally:
        env.close()
    environment_elapsed = time.monotonic() - environment_started
    versions = torch.zeros(1, dtype=torch.int32)
    server = ParameterServer(0, versions, False)
    mode = getattr(cfg, "controller_learning", "ppo")
    learner = make_hipposlam_learner(cfg, info, versions, 0, server)
    initialization_started = time.monotonic()
    learner.init()
    initialization_elapsed = time.monotonic() - initialization_started
    saved = load_checkpoint_dict(immutable, torch.device("cpu"))
    assert_exact(saved["model"], learner.actor_critic.state_dict())
    optimizer_contract = assert_optimizer_restore(
        saved["optimizer"], learner.optimizer.state_dict(), saved.get("curr_lr", cfg.learning_rate), learner.curr_lr
    )
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
        protocol="learner-checkpoint-restore-v2",
        run=cfg.experiment,
        checkpoint=str(immutable),
        checkpoint_sha256=digest.hexdigest(),
        frames=learner.env_steps,
        train_step=learner.train_step,
        controller=mode,
        exact_restore=optimizer_contract["optimizer_exact"],
        restoration_verified=True,
        restore_contract="learner_init_scheduled_lr_v1",
        model_and_buffers_exact=True,
        **optimizer_contract,
        counters_exact=True,
        device=str(learner.device),
        job_id=os.environ.get("SLURM_JOB_ID"),
        timing_seconds=dict(
            environment_setup=environment_elapsed, learner_init=initialization_elapsed, total=time.monotonic() - started
        ),
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
