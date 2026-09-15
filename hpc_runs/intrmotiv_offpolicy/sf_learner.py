"""Native SF learner hook retaining v2 recurrent DDQN/HER contracts."""

import json
import time
from pathlib import Path

import torch

from sample_factory.algo.learning.learner import BaseLearner, model_initialization_data
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, TRAIN_STATS
from sample_factory.algo.utils.shared_buffers import policy_device
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.utils import experiment_dir

from .batch import PositionBatcher, learn_batch
from .checkpoint import state_hash
from .replay import Observation, SequenceReplay, Transition

# Import in spawned workers before Batcher.init allocates its training buffers.
from .sf_buffers import install
from .sf_transport import OrderedIngress, updates_due
from .telemetry import Coverage
from .train import json_write, save_checkpoint
from .worker import DoubleDQNLearner

install()


class NativeLearner(BaseLearner):
    def init(self):
        torch.set_num_threads(self.cfg.ddqn_learner_threads)
        torch.manual_seed(self.cfg.seed)
        self.device = policy_device(self.cfg, self.policy_id)
        self.actor_critic = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
        self.actor_critic.model_to_device(self.device)
        torch.set_num_threads(self.cfg.ddqn_learner_threads)
        self.args = self.cfg.ddqn_args
        self.learner = DoubleDQNLearner(
            self.actor_critic.worker, target_period=self.args["target_period"], execution=self.args["learner_execution"]
        )
        if not self.cfg.serial_mode:
            self.actor_critic._apply(lambda t: t.share_memory_() if not t.is_cuda else t)
        self.param_server.init(self.actor_critic, 0, self.device)
        self.reference_hash = state_hash(self.actor_critic.parent.encoder)
        self.replay = SequenceReplay(
            self.args["replay_capacity"], self.learner.online.width, self.cfg.seed, self.reference_hash
        )
        # Buffer reuse may reorder delivery. Bound waiting storage independently of replay.
        self.ingress = OrderedIngress(self.cfg.ddqn_max_pending)
        self.batcher = PositionBatcher()
        self.coverage = Coverage(self.cfg.ddqn_registry)
        self.output = Path(experiment_dir(self.cfg))
        self.output.mkdir(parents=True, exist_ok=True)
        self.log_file = (self.output / "metrics.jsonl").open("x", buffering=1)
        self.accepted = self.invalid = self.decisions = self.positions = self.her_positions = 0
        self.samples = self.her_samples = self.attempts = self.arrivals = 0
        self.last_metrics = {}
        self.telemetry = None
        if getattr(self.cfg, "ddqn_telemetry", False):
            from .sf_telemetry import NativeTelemetry

            self.telemetry = NativeTelemetry(self.cfg, self.env_info, self.policy_id, self.actor_critic)
        self.ingestion_seconds = self.learning_seconds = 0.0
        self.started = self.last_log = time.monotonic()
        self.finished = False
        self.milestones = sorted(set(int(x) for x in self.args.get("milestones", "0").split(",")))
        self.is_initialized = True
        json_write(self.output / "conversion.json", self.actor_critic.report)
        json_write(
            self.output / "run_config.json",
            dict(
                schema="intrmotiv/ddqn-run/v2",
                args=self.args,
                parent=self.cfg.ddqn_parent,
                representation="frozen_source",
                exploration="epsilon_greedy_fixed_command",
                control="first_arrival",
                execution="sample_factory_native",
                graph_learning=False,
                pose_input=False,
                transport="cached_actor_features_v1",
                restart_kind="warm_restart_requires_refill",
            ),
        )
        self.save()
        return model_initialization_data(self.cfg, self.policy_id, self.actor_critic, 0, self.device)

    def _ingest(self, batch):
        n, bypass = self.learner.online.n_goals, self.learner.online.bypass_size
        # One transfer per dense array; ingestion owns all pending feature rows.
        packets = batch["ddqn_packet"].detach().cpu()
        identities = batch["obs"]["ddqn_identity"][:, :-1].detach().cpu()
        actions = batch["actions"].detach().cpu().long().squeeze(-1)
        dones = batch["dones"].detach().cpu()
        timeouts = batch["time_outs"].detach().cpu()
        versions = batch["policy_version"].detach().cpu().long()
        if not torch.all(batch["policy_id"] == self.policy_id):
            raise ValueError("unexpected multi-policy data")
        for b in range(packets.shape[0]):
            for t in range(packets.shape[1]):
                p = packets[b, t]
                stream, episode, index, serial = map(int, identities[b, t])
                goal, budget, option = map(int, p[-3:])
                observation = Observation(p[:n], p[n : n + bypass], p[n + bypass : n + bypass + n].bool()).owned()
                row = Transition(
                    stream,
                    episode,
                    index,
                    observation,
                    None,
                    int(actions[b, t]),
                    goal,
                    option,
                    budget,
                    bool(dones[b, t] and not timeouts[b, t]),
                    bool(timeouts[b, t]),
                    False,
                    int(versions[b, t]),
                )
                self.ingress.add(stream, serial, row)
        for row in self.ingress.drain():
            self.replay.append(row)
            self.accepted += int(row.successor_valid)
            self.invalid += int(not row.successor_valid)
            self.decisions += 1
            hit = row.successor_valid and bool(row.successor.events[row.goal])
            ended = hit or row.budget <= 1 or row.terminated or row.truncated
            self.coverage.collect(row.stream, row.episode, row.successor, row.goal, ended)
            self.attempts += int(ended)
            self.arrivals += int(hit)
            if self.decisions * self.env_info.frameskip >= self.args["total_frames"]:
                self.finished = True
                break
        self.env_steps = self.decisions * self.env_info.frameskip

    def _learn(self):
        for _ in range(
            updates_due(
                self.accepted, self.args["learning_start"], self.args["decisions_per_update"], self.learner.updates
            )
        ):
            try:
                samples = self.batcher.sample(
                    self.replay, self.cfg.ddqn_registry, self.args["her_fraction"], self.args["td_positions_per_update"]
                )
            except ValueError:
                # Retain debt until an eligible physical prefix/future exists.
                break
            with self.param_server.policy_lock:
                self.last_metrics = learn_batch(self.learner, samples, self.device)
            self.coverage.replay(samples)
            self.samples += len(samples)
            self.her_samples += sum(s["relabeled"] for s in samples)
            self.positions += self.last_metrics["valid_loss_positions"]
            self.her_positions += sum(int(s["mask"].sum()) for s in samples if s["relabeled"])
        self.train_step = self.learner.updates
        self.param_server.update_weights(self.train_step)

    def metrics(self):
        return dict(
            frames=self.env_steps,
            decisions=self.decisions,
            accepted=self.accepted,
            updates=self.train_step,
            invalid_final=self.invalid,
            attempts=self.attempts,
            arrivals=self.arrivals,
            throughput_fps=self.env_steps / max(1e-9, time.monotonic() - self.started),
            requested_her_fraction=self.args["her_fraction"],
            target_period_updates=self.args["target_period"],
            decisions_per_update=self.args["decisions_per_update"],
            td_positions_per_update=self.args["td_positions_per_update"],
            effective_loss_positions_per_decision=self.positions / max(1, self.decisions),
            inference_decisions=int(self.actor_critic.counter.item()),
            epsilon=max(0.1, 1 - 0.9 * int(self.actor_critic.counter.item()) / 250000),
            realized_her_fraction=self.her_samples / max(1, self.samples),
            valid_loss_positions_total=self.positions,
            her_loss_positions=self.her_positions,
            original_loss_positions=self.positions - self.her_positions,
            ingestion_seconds=self.ingestion_seconds,
            learning_seconds=self.learning_seconds,
            transport_received=self.ingress.received,
            transport_pending=self.ingress.pending,
            update_debt=updates_due(
                self.accepted, self.args["learning_start"], self.args["decisions_per_update"], self.train_step
            ),
            replay_size=len(self.replay.rows),
            **self.coverage.metrics(),
            **self.last_metrics,
        )

    def train(self, batch):
        if self.finished:
            return {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        report = False
        telemetry_stats = {}
        if not self.finished:
            before = self.env_steps
            start = time.monotonic()
            self._ingest(batch)
            if self.telemetry is not None:
                telemetry_stats = self.telemetry.capture(batch, self.policy_id, self.env_steps)
            self.ingestion_seconds += time.monotonic() - start
            start = time.monotonic()
            self._learn()
            self.learning_seconds += time.monotonic() - start
            now = time.monotonic()
            if now - self.last_log >= 10 or self.finished:
                metrics = self.metrics()
                self.log_file.write(json.dumps(metrics) + "\n")
                self.last_log = now
                report = True
            if not self.finished and any(before < m <= self.env_steps for m in self.milestones):
                self.save_milestone()
            if self.finished:
                if self.metrics()["update_debt"]:
                    raise RuntimeError("final exact update budget has outstanding debt")
                self.save()
        metrics = self.metrics()
        result = {LEARNER_ENV_STEPS: self.env_steps, POLICY_ID_KEY: self.policy_id}
        if report:
            result.update({TRAIN_STATS: dict(self.last_metrics), "ddqn_metrics": metrics})
        if telemetry_stats:
            result.setdefault(TRAIN_STATS, {}).update(telemetry_stats)
        return result

    def save(self):
        if not self.is_initialized:
            return False
        save_checkpoint(
            self.output, self.learner, self.replay, self.env_steps, self.decisions, self.args, self.reference_hash
        )
        if self.finished:
            unchanged = state_hash(self.actor_critic.parent.encoder) == self.reference_hash
            if not unchanged:
                raise RuntimeError("frozen reference changed")
            json_write(
                self.output / "runtime_gate.json",
                dict(
                    frames=self.env_steps,
                    updates=self.train_step,
                    target_copies=self.train_step // self.learner.target_period,
                    invalid_final_exclusions=self.invalid,
                    frozen_reference_unchanged=unchanged,
                    her_samples=self.her_samples,
                    target_period_updates=self.learner.target_period,
                    valid_loss_positions=self.positions,
                    td_positions_per_update=self.args["td_positions_per_update"],
                    accepted=self.accepted,
                    decisions=self.decisions,
                    update_debt=self.metrics()["update_debt"],
                    transport_received=self.ingress.received,
                    transport_pending=self.ingress.pending,
                    transport_emitted=self.ingress.emitted,
                    pending_loss_positions=len(self.batcher.pending["segment"]) if self.batcher.pending else 0,
                    execution="sample_factory_native",
                    scientific_qualification="pending_independent_commanded_evaluation",
                ),
            )
        return True

    def save_milestone(self):
        self.save()

    def save_best(self, *args):
        return False

    def set_new_cfg(self, cfg):
        raise ValueError("PBT is not qualified for recurrent DDQN/HER")

    def set_policy_to_load(self, policy_id):
        raise ValueError("cross-policy loading is not qualified")


def make_native_learner(*args, **kwargs):
    return NativeLearner(*args, **kwargs)
