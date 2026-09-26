"""Paired learner microbenchmark using the actual SF TargetFiLMDecoder.

No environment rollout and no scientific training result. Identical replay
batches, initialization, TD budgets and optimizer counts across backends.
"""

import argparse
import copy
import hashlib
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .batch import PositionBatcher, learn_batch
from .contracts import canonical_events
from .replay import Observation, SequenceReplay, Transition
from .worker import DoubleDQNLearner, QWorker


def make_worker(checkpoint):
    from sf_working_directories.IntrMotiv.dmlab.custom_actor_critic import TargetFiLMDecoder

    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved["schema"] != QWorker.schema:
        raise ValueError("benchmark requires a compatible v2 checkpoint")
    state = saved["model"]
    n = state["decoder.target_modulation"].shape[0]
    hidden = state["decoder.state_layer.0.weight"].shape[0]
    input_size = state["decoder.state_layer.0.weight"].shape[1]
    # The qualified frozen F16 parent contract is R=8,L=64.
    width = 71
    bypass = input_size - n * width
    core = SimpleNamespace(Hippo_n_feature=n, target_condition_start=input_size, get_out_size=lambda: input_size + n)
    decoder = TargetFiLMDecoder(core, hidden_size=hidden)
    worker = QWorker(decoder, n, bypass, repeat_width=8, length=64, batch_independent=True)
    worker.load_state_dict(state, strict=True)
    return worker


def make_batches(worker, fraction, count=8):
    rng = torch.Generator().manual_seed(391)
    replay = SequenceReplay(4096, worker.width, seed=17, reference_hash="synthetic-profile-only")
    for stream in range(16):
        pre = torch.randn(129, worker.n_goals, generator=rng) + 1.1
        bypass = torch.randn(129, worker.bypass_size, generator=rng)
        events = canonical_events(torch.relu(pre - worker.intercept), exclusive=True)
        obs = [Observation(p, b, e) for p, b, e in zip(pre, bypass, events)]
        for i in range(128):
            replay.append(
                Transition(
                    stream,
                    0,
                    i,
                    obs[i],
                    obs[i + 1],
                    i % 8,
                    (1, 4, 11)[stream % 3],
                    i // 64,
                    64 - i % 64,
                    terminated=i == 127,
                )
            )
    batcher = PositionBatcher()
    return [batcher.sample(replay, [1, 4, 11], fraction, 256) for _ in range(count)]


def parity(worker, batches):
    a = DoubleDQNLearner(copy.deepcopy(worker), target_period=2)
    b = DoubleDQNLearner(copy.deepcopy(worker), target_period=2, execution="batched")
    loss_error = weight_error = 0.0
    for samples in batches[:3]:
        x = learn_batch(a, samples, "cpu")
        y = learn_batch(b, samples, "cpu")
        loss_error = max(loss_error, abs(x["td_loss"] - y["td_loss"]))
        assert x["valid_loss_positions"] == y["valid_loss_positions"] == 256
        assert x["target_copies"] == y["target_copies"]
        for p, q in zip(a.online.parameters(), b.online.parameters()):
            weight_error = max(weight_error, float((p - q).detach().abs().max()))
            torch.testing.assert_close(p, q, rtol=1e-4, atol=1e-5)
    assert loss_error < 1e-5
    return dict(
        max_loss_difference=loss_error,
        max_parameter_difference=weight_error,
        updates=3,
        valid_td_positions=768,
        target_copies=1,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if min(args.steps, args.repeats) < 1:
        p.error("positive timing budgets required")
    torch.set_num_threads(1)
    worker = make_worker(args.checkpoint)
    fixtures = {name: make_batches(worker, fraction) for name, fraction in [("DDQN", 0.0), ("DDQN_HER", 0.8)]}
    result = dict(
        scope="Synthetic cached-feature learner benchmark; no DMLab FPS claim",
        checkpoint=str(args.checkpoint),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        decoder=type(worker.decoder).__name__,
        shape=dict(goals=worker.n_goals, width=worker.width, bypass=worker.bypass_size),
        torch_version=torch.__version__,
        threads=args.threads,
        steps=args.steps,
        repeats=args.repeats,
        parity={},
        measurements=[],
    )
    for threads in map(int, args.threads.split(",")):
        torch.set_num_threads(threads)
        result["parity"][str(threads)] = {name: parity(worker, batches) for name, batches in fixtures.items()}
        for name, batches in fixtures.items():
            for backend in ("reference", "batched"):
                timings = []
                prefix = []
                learner_time = []
                for repeat in range(args.repeats):
                    learner = DoubleDQNLearner(copy.deepcopy(worker), target_period=100, execution=backend)
                    for i in range(5):
                        learn_batch(learner, batches[i % len(batches)], "cpu")
                    begin = time.perf_counter()
                    for i in range(args.steps):
                        m = learn_batch(learner, batches[i % len(batches)], "cpu")
                        prefix.append(m["prefix_and_batch_seconds"])
                        learner_time.append(m["learner_update_seconds"])
                    timings.append((time.perf_counter() - begin) / args.steps)
                seconds = statistics.median(timings)
                row = dict(
                    arm=name,
                    backend=backend,
                    threads=threads,
                    seconds_per_update=seconds,
                    valid_td_positions_per_update=256,
                    td_positions_per_second=256 / seconds,
                    prefix_batch_ms=statistics.mean(prefix) * 1000,
                    learner_ms=statistics.mean(learner_time) * 1000,
                    repetition_seconds_per_update=timings,
                )
                result["measurements"].append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
