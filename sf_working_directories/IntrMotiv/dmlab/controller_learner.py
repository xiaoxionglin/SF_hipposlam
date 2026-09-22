"""Original fresh DG learner followed by separately budgeted controller replay."""

import copy
import random
import time
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sample_factory.algo.learning.learner import model_initialization_data
from sample_factory.algo.utils.misc import LEARNER_ENV_STEPS, POLICY_ID_KEY, TRAIN_STATS
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.attr_dict import AttrDict

from .ca3_state_readout import paired_anchor_improvement
from .controller_history import ControllerHistory, reconstruct_controller_histories, reconstruct_controller_history
from .controller_q import continuing_double_q_target
from .controller_replay import PhysicalDecision, PhysicalReplay
from .controller_schedule import UpdateClock, replay_rngs
from .controller_snapshot import ControllerSnapshot, differentiable_replay, evaluate_replay, fresh_dg_parameter_owner
from .controller_transition import ReplayRejected, TransitionInput, transition_values, transition_values_batch
from .controller_transport import IDENTITY_KEY, cached_observation
from .custom_learner import DistanceLearnerReward
from .hrl_controllable_graph import HRLStateLayout, current_dg_from_activity


class ControllerLearner(DistanceLearnerReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        c = self.cfg
        self.replay = PhysicalReplay(c.controller_replay_capacity, c.seed)
        self.replay.rng, self.her_rng = replay_rngs(c.seed)
        self.clock = UpdateClock(
            c.controller_learning_starts, c.controller_decisions_per_update, c.controller_target_updates
        )
        self.controller_version = 0
        self.controller_stats = {}
        self.frame_milestones = set()
        self.fresh_dg_steps = self.fresh_graph_batches = 0
        self._restore_controller = None
        self.publication = 0
        self._milestone_previous = None

    def init(self):
        super().init()

        def factory():
            model = create_actor_critic(self.cfg, self.env_info.obs_space, self.env_info.action_space)
            model.model_to_device(self.device)
            return model

        self.online_snapshot = ControllerSnapshot(factory, self.actor_critic, self.controller_version)
        self.target_snapshot = ControllerSnapshot(factory, self.actor_critic, self.controller_version)
        self.published = ControllerSnapshot(factory, self.actor_critic, self.controller_version).model
        self.published._apply(lambda t: t.share_memory_() if not t.is_cuda else t)
        if self._restore_controller is not None:
            c = self._restore_controller
            self.replay.load_state_dict(c["replay"])
            self.clock.load_state_dict(c["clock"])
            self.frame_milestones = set(c.get("frame_milestones", ()))
            self.her_rng.bit_generator.state = c["her_rng"]
            self.controller_version = c["version"]
            self.publication = c["publication"]
            self.target_snapshot.model.load_state_dict(c["target"])
            self.target_snapshot.version = c["target_version"]
            self.online_snapshot.version = self.controller_version
            self.fresh_dg_steps = c["fresh_dg_steps"]
            self.fresh_graph_batches = c["fresh_graph_batches"]
            torch.set_rng_state(c["torch_rng"].cpu())
            np.random.set_state(
                (c["numpy_rng"][0], c["numpy_rng"][1].cpu().numpy().astype(np.uint32), *c["numpy_rng"][2:])
            )
            random.setstate(c["python_rng"])
            if torch.cuda.is_available() and c["cuda_rng"]:
                torch.cuda.set_rng_state_all([rng.cpu() for rng in c["cuda_rng"]])
        # Actors only see this copy. The parent's minibatch optimizer/publication
        # writes affect private learner storage until the transaction finishes.
        self._published_versions = self.policy_versions_tensor
        self.policy_versions_tensor = self.policy_versions_tensor.clone()
        self.param_server.init(self.published, self.publication, self.device)
        if getattr(self.cfg, "save_initial_checkpoint", False) and self.env_steps == 0:
            self._save_impl("initial", "", 1)
        return model_initialization_data(self.cfg, self.policy_id, self.published, self.publication, self.device)

    def _save_impl(self, *args, **kwargs):
        if not hasattr(self, "target_snapshot"):
            return False
        return super()._save_impl(*args, **kwargs)

    def save_milestone(self):
        # Keep complete restart state, including replay. Bound periodic archives
        # by SF's existing retention setting, while pinning canonical targets.
        super().save_milestone()
        directory = Path(self.checkpoint_dir(self.cfg, self.policy_id)) / "milestones"
        files = [Path(path) for path in self.get_checkpoints(str(directory)) if Path(path).suffix == ".pth"]
        keep = max(1, int(self.cfg.keep_checkpoints))
        retained = {path.name for path in files[-keep:]} | self.frame_milestones
        for path in files:
            if path.name not in retained:
                path.unlink()

    def _save_completed_frame_targets(self, previous_frames):
        targets = [int(x) for x in self.cfg.checkpoint_frame_targets.split(",") if x.strip()]
        if any(previous_frames < target <= self.env_steps for target in targets):
            self.frame_milestones.add(f"checkpoint_{self.train_step:09d}_{self.env_steps}.pth")
        super()._save_crossed_frame_targets(previous_frames)

    def _save_crossed_frame_targets(self, previous_frames):
        # The parent invokes this before replay. Save only after publication.
        self._milestone_previous = previous_frames

    def _load_state(self, checkpoint_dict, load_progress=True):
        super()._load_state(checkpoint_dict, load_progress)
        if load_progress:
            if "controller" not in checkpoint_dict:
                raise RuntimeError("Controller resume requires complete replay/target state")
            self._restore_controller = checkpoint_dict["controller"]
            if self._restore_controller.get("replay_state", "reconstruct") != getattr(
                self.cfg, "controller_replay_state", "reconstruct"
            ):
                raise RuntimeError("Replay state contract changed; start a fresh run")

    def _optimizer_step_counts(self):
        groups = {"dg": self.actor_critic.encoder.DG_projection, "main": self.actor_critic.controller_q.main}
        return {
            group: {
                name: int(self.optimizer.state.get(parameter, {}).get("step", 0))
                for name, parameter in module.named_parameters()
            }
            for group, module in groups.items()
        }

    def _get_checkpoint_dict(self):
        checkpoint = super()._get_checkpoint_dict()
        if hasattr(self, "target_snapshot"):
            checkpoint["controller"] = dict(
                replay_state=getattr(self.cfg, "controller_replay_state", "reconstruct"),
                optimizer_step_counts=self._optimizer_step_counts(),
                replay=self.replay.state_dict(),
                clock=self.clock.state_dict(),
                frame_milestones=sorted(self.frame_milestones),
                her_rng=copy.deepcopy(self.her_rng.bit_generator.state),
                version=self.controller_version,
                publication=self.publication,
                target=self.target_snapshot.model.state_dict(),
                target_version=self.target_snapshot.version,
                fresh_dg_steps=self.fresh_dg_steps,
                fresh_graph_batches=self.fresh_graph_batches,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                numpy_rng=(
                    np.random.get_state()[0],
                    torch.as_tensor(np.random.get_state()[1].astype(np.int64)),
                    *np.random.get_state()[2:],
                ),
                python_rng=random.getstate(),
            )
        return checkpoint

    def _prepare_batch(self, batch):
        buff, size, invalid = super()._prepare_batch(batch)
        if invalid < size:
            identities = batch["obs"][IDENTITY_KEY][:, :-1].reshape(-1, 4).cpu().tolist()
            labels = (
                "hrl_control_correct_outcome",
                "hrl_control_wrong_outcome",
                "hrl_control_target_timeout",
                "hrl_control_exploration_timeout",
                "hrl_control_outcome_id",
                "hrl_control_reward_magnitude",
            )
            for index, (stream, episode, decision, serial) in enumerate(identities):
                key = ((self.replay.session, int(stream)), int(episode), int(decision))
                self.replay.annotate(
                    key, buff["rewards"][index], {name: float(buff[name][index]) for name in labels if name in buff}
                )
        return buff, size, invalid

    def _prepare_recurrent_replay_head(self, head_outputs, mb):
        if self.cfg.controller_learning == "ddqn" and getattr(self.cfg, "dg_goal_input", "none") == "write":
            return torch.cat((head_outputs, mb.controller_condition), -1)
        return super()._prepare_recurrent_replay_head(head_outputs, mb)

    def _override_core_outputs_for_replay(self, core_outputs, mb):
        if self.cfg.controller_learning == "ddqn":
            out = core_outputs.clone()
            core = self.actor_critic.core
            out[:, core.target_condition_start : core.total_output_size] = mb.controller_condition
            return out
        return super()._override_core_outputs_for_replay(core_outputs, mb)

    def _update_policy_graph_from_rollout(self, *args, **kwargs):
        result = super()._update_policy_graph_from_rollout(*args, **kwargs)
        self.fresh_graph_batches += 1
        return result

    def _after_optimizer_step(self):
        super()._after_optimizer_step()
        self.fresh_dg_steps += 1
        self._update_contextual_anchors()

    @torch.no_grad()
    def _update_contextual_anchors(self):
        """Consume maximal exclusive-DG occurrences after W has been updated."""
        pending = getattr(self, "_pending_contextual_anchors", None)
        self._pending_contextual_anchors = None
        if pending is None:
            return
        flat_states, flat_actions, flat_valids, flat_dones, flat_previous, recurrence = pending
        core = self.actor_critic.core
        graph = core.policy_graph
        if graph is None or not graph.contextual:
            return
        n, expanded = graph.n_nodes, core.expanded_length
        streams = flat_states.size(0) // recurrence
        states = flat_states.reshape(streams, recurrence, -1)
        dg = states.reshape(streams, recurrence, n, expanded)[..., 0]
        actions = flat_actions.reshape(streams, recurrence, -1)[..., 0].long()
        valid = flat_valids.reshape(streams, recurrence).bool()
        done = flat_dones.reshape(streams, recurrence).bool()
        previous = flat_previous.reshape(streams, recurrence, -1)[:, 0]
        previous_dg = previous.reshape(streams, n, expanded)[..., 0]
        horizon = int(self.cfg.ca3_state_readout_horizon)
        decision = int(getattr(self.replay, "accepted", self.env_steps))
        runs = []
        for stream in range(streams):
            t = 0
            while t < recurrence:
                active = torch.where(dg[stream, t] > 0)[0] if valid[stream, t] else torch.empty(0, device=dg.device)
                if active.numel() != 1:
                    t += 1
                    continue
                node, start = int(active[0]), t
                t += 1
                while t < recurrence and valid[stream, t] and not done[stream, t - 1]:
                    next_active = torch.where(dg[stream, t] > 0)[0]
                    if next_active.numel() != 1 or int(next_active[0]) != node:
                        break
                    t += 1
                if start == 0:
                    prior_active = torch.where(previous_dg[stream] > 0)[0]
                    if prior_active.numel() == 1 and int(prior_active[0]) == node:
                        # This chunk begins inside a maximal occurrence; only
                        # its true onset may register or confirm an anchor.
                        continue
                runs.append((stream, start, t, node))
        for stream, start, end, node in runs:
            if end - start > 1:
                for offset in range(start + 1, end):
                    pair_complete = offset + horizon < recurrence
                    if pair_complete:
                        pair_complete = bool(valid[stream, offset : offset + horizon + 1].all()) and not bool(
                            done[stream, offset : offset + horizon].any()
                        )
                    if pair_complete:
                        graph.add_positive_pair(
                            states[stream, start],
                            states[stream, offset],
                            actions[stream, offset : offset + horizon],
                            dg[stream, offset + 1 : offset + horizon + 1],
                        )
        calibration_started = time.perf_counter()
        calibrated = graph.recalibrate(
            core.state_readout,
            core.innovation_predictor,
            decision,
            int(self.cfg.ca3_context_calibration_interval),
            int(self.cfg.ca3_context_calibration_min_pairs),
            float(self.cfg.ca3_context_calibration_quantile),
            float(self.cfg.ca3_state_readout_active_coeff),
            float(self.cfg.ca3_state_readout_zero_coeff),
        )
        if calibrated:
            self.controller_stats["calibration_seconds"] = time.perf_counter() - calibration_started
        for stream, start, _end, node in runs:
            complete = start + horizon < recurrence
            if complete:
                complete = bool(valid[stream, start : start + horizon + 1].all()) and not bool(
                    done[stream, start : start + horizon].any()
                )
            candidate = states[stream, start]
            if not graph.anchor_valid[node]:
                if complete:
                    graph.register_anchor(node, candidate, decision)
                continue
            if not complete:
                continue
            future = dg[stream, start + 1 : start + horizon + 1]
            action_window = actions[stream, start : start + horizon]
            graph.add_diagnostic_pair(graph.anchor_ca3[node], candidate)
            if complete and getattr(self.cfg, "ca3_graph_anchor_mode", "fixed") == "champion":
                mean, lower, score = paired_anchor_improvement(
                    core.state_readout,
                    core.innovation_predictor,
                    graph.anchor_ca3[node],
                    candidate,
                    action_window,
                    future,
                    float(self.cfg.ca3_state_readout_active_coeff),
                    float(self.cfg.ca3_state_readout_zero_coeff),
                )
                if lower > 0:
                    graph.replace_anchor(node, candidate, decision, float(score))
                    continue
            confirmed = graph.confirm_anchor(
                node,
                candidate,
                core.state_readout,
                core.innovation_predictor,
                action_window,
                future,
                float(self.cfg.ca3_state_readout_active_coeff),
                float(self.cfg.ca3_state_readout_zero_coeff),
                decision,
            )
            if confirmed and getattr(self.cfg, "ca3_graph_anchor_mode", "fixed") == "ema":
                graph.refine_anchor_ema(
                    node,
                    candidate,
                    core.state_readout,
                    core.innovation_predictor,
                    decision,
                    float(self.cfg.ca3_graph_anchor_ema_alpha),
                    int(self.cfg.ca3_graph_anchor_ema_min_confirmations),
                    float(self.cfg.ca3_graph_anchor_ema_margin),
                )

    def _calculate_losses(self, mb, num_invalids, iterative_phase, **kwargs):
        if self.cfg.controller_learning != "ddqn":
            return super()._calculate_losses(mb, num_invalids, iterative_phase, **kwargs)
        additional = AttrDict()
        valids = mb.valids
        recurrence = self.cfg.recurrence
        outputs = self._forward_fresh_dg(mb, recurrence, valids, iterative_phase, additional)
        distance, masked, _ = self._record_distance_matrix(
            outputs.core_outputs.detach(),
            minibatch_size=outputs.minibatch_size,
            masked_matrix=True,
            return_progression=True,
        )
        additional["Distance Matrix"] = distance
        additional["Distance Matrix Masked"] = masked
        encoder_loss = self._calculate_fresh_encoder_loss(outputs, mb, valids, num_invalids, recurrence, additional)
        zero = encoder_loss.new_zeros(())
        # Retain the parent's dashboard contract without computing PPO losses.
        for name in (
            "behavior_replay_mismatch",
            "ca3_predictor_loss",
            "ca3_predictor_hit_accuracy",
            "ca3_predictor_time_mae",
            "ca3_predictor_positive_fraction",
            "empirical_her_loss",
            "empirical_her_policy_loss",
            "empirical_her_value_loss",
            "empirical_her_ratio",
            "empirical_her_clip_fraction",
            "empirical_her_valid_fraction",
            "goal_condition_target_valid_fraction",
            "goal_condition_action_sensitivity",
            "goal_condition_action_probability_tv",
            "goal_condition_value_span",
        ):
            additional[name] = zero
        if kwargs.get("record_goal_diagnostics", True):
            self._record_goal_condition_diagnostics(outputs, mb, additional)
        for mode in ("goal", "free"):
            for name in ("policy_loss", "value_loss", "entropy_loss"):
                additional[f"{mode}_{name}"] = zero
        summaries = dict(
            ratio=torch.ones_like(mb.advantages),
            clip_ratio_low=1.0,
            clip_ratio_high=1.0,
            values=outputs.result["values"],
            adv=mb.advantages,
            adv_std=zero,
            adv_mean=zero,
            additional_stats=additional,
        )
        return (
            self.actor_critic.action_distribution(),
            zero,
            zero,
            zero[None],
            zero,
            zero,
            zero,
            encoder_loss,
            summaries,
        )

    def _ingest(self, batch):
        # Shared SF buffers may be reused immediately after this transaction.
        def cpu(x):
            return x.detach().cpu().numpy().copy()

        n, t = batch["actions"].shape[:2]
        layout = HRLStateLayout(self.cfg.Hippo_n_feature)
        stored = getattr(self.cfg, "controller_replay_state", "reconstruct") == "stored"
        terminal_labels = {}
        if stored:
            from .controller_history import reconstruction_head

            mask = batch["dones"].bool() & batch["controller_final_valid"].bool()
            if mask.any():
                coordinates = mask.nonzero().cpu().tolist()
                observations = {k: v[mask] for k, v in batch["controller_final_obs"].items() if k != IDENTITY_KEY}
                with torch.no_grad():
                    dg = reconstruction_head(self.published, observations)[:, : self.cfg.Hippo_n_feature].cpu().numpy()
                terminal_labels = {tuple(key): value.copy() for key, value in zip(coordinates, dg)}
        for i in range(n):
            for j in range(t):
                stream, episode, index, serial = map(int, cpu(batch["obs"][IDENTITY_KEY][i, j]))
                obs = {} if stored else {k: cpu(v[i, j]) for k, v in batch["obs"].items() if k != IDENTITY_KEY}
                if not stored and self.cfg.controller_cache_visual:
                    obs = cached_observation(obs, cpu(batch["controller_visual"][i, j]))
                done = bool(batch["dones"][i, j])
                valid = bool(batch["controller_final_valid"][i, j])
                successor = None
                if done and valid and not stored:
                    successor = {k: cpu(v[i, j]) for k, v in batch["controller_final_obs"].items() if k != IDENTITY_KEY}
                    # Terminal RGB is retained. It is converted once when sampled.
                context = cpu(batch["controller_context"][i, j])
                self.replay.receive(
                    PhysicalDecision(
                        (self.replay.session, stream),
                        episode,
                        index,
                        serial,
                        obs,
                        int(batch["actions"][i, j].item()),
                        cpu(batch["controller_condition"][i, j]),
                        context,
                        int(batch["policy_version"][i, j]),
                        int(context[layout.persistent_start]),
                        bool(batch["controller_terminated"][i, j]),
                        bool(batch["time_outs"][i, j]),
                        int(batch["controller_frames"][i, j]),
                        successor,
                        valid,
                        worker_state=cpu(batch["controller_worker_state"][i, j]) if stored else None,
                        terminal_dg=terminal_labels.get((i, j)),
                        terminal_publication=self.publication if (i, j) in terminal_labels else None,
                        anchor_generation=int(batch["controller_anchor_generation"][i, j].item()),
                    )
                )
        self.controller_stats["actor_memory_rebuilds"] = float(batch["controller_memory_stats"][..., 0].max())
        self.controller_stats["actor_memory_rebuild_seconds"] = float(batch["controller_memory_stats"][..., 2].max())
        self.controller_stats["actor_memory_version_failures"] = float(batch["controller_memory_stats"][..., 1].max())

    def _example(self, key, allow_stale_anchor=False):
        if getattr(self.cfg, "controller_replay_state", "reconstruct") == "stored":
            from .controller_stored_replay import example_from_replay

            return example_from_replay(self, key, allow_stale_anchor=allow_stale_anchor)
        prefix, suffix = self.replay.sequence(key, self.actor_critic.core.expanded_length, 2)
        row = suffix[0]
        generation = int(self.actor_critic.core.policy_graph.representation_generation.item())
        if any(r.generation != generation for r in prefix + suffix):
            raise ReplayRejected("stale_structural_generation")
        if row.terminated or row.truncated:
            if not row.successor_valid:
                raise ReplayRejected("uncertified_terminal_observation")
            obs = row.successor
            if self.cfg.controller_cache_visual:
                from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs

                raw = {k: torch.as_tensor(v, device=self.device)[None] for k, v in obs.items()}
                with torch.no_grad():
                    norm = prepare_and_normalize_obs(self.online_snapshot.model, raw)
                    self.online_snapshot.model.forward_head(norm)
                    visual = self.online_snapshot.model.encoder._controller_visual[0].cpu().numpy()
                obs = cached_observation(obs, visual)
            successor = replace(row, index=row.index + 1, serial=row.serial + 1, observation=obs)
        elif len(suffix) < 2:
            raise ReplayRejected("missing_successor_context")
        else:
            successor = suffix[1]
        return TransitionInput(tuple(prefix + [row, successor]), len(prefix))

    def set_progress_callback(self, callback):
        self._progress_callback = callback
        self._last_progress_heartbeat = time.monotonic()

    def _report_controller_progress(self):
        # Reuse SF's watchdog signal only after actual work has completed.
        # A blocked GPU operation still fails the ordinary heartbeat timeout.
        callback = getattr(self, "_progress_callback", None)
        if callback is not None:
            now = time.monotonic()
            if now - self._last_progress_heartbeat >= self.cfg.heartbeat_interval:
                callback()
                self._last_progress_heartbeat = now

    def _reconstruction_width(self):
        return min(self.cfg.controller_td_positions, 256 if self.device.type == "cuda" else 16)

    def _evaluate_pairs(self, examples):
        if getattr(self.cfg, "controller_replay_state", "reconstruct") == "stored":
            from .controller_stored_replay import evaluate_pairs

            return evaluate_pairs(self, examples)
        if not examples:
            return []
        # This is a memory bound, not a sampling or optimizer batch size. All
        # accepted positions still contribute to one mean and one optimizer step.
        width = self._reconstruction_width()
        if len(examples) > width:
            return [
                result
                for start in range(0, len(examples), width)
                for result in self._evaluate_pairs(examples[start : start + width])
            ]
        # Screening only changes eligibility-search work. Keep the original
        # batch geometry for all main/HER gradient and target-value evaluations.
        operation = partial(transition_values_batch, screen_current=not torch.is_grad_enabled())
        online = differentiable_replay(
            self.online_snapshot, self.actor_critic, operation, examples, source_version=self.controller_version
        )
        target = evaluate_replay(self.target_snapshot, operation, examples)
        results = []
        for example, a, b in zip(examples, online, target):
            if isinstance(a, str) or isinstance(b, str):
                results.append(a if isinstance(a, str) else b)
                continue
            if not torch.equal(a["signature"], b["signature"]):
                results.append("target_event_incompatible")
                continue
            if a["done"] != b["done"]:
                results.append("target_boundary_incompatible")
                continue
            row = example.rows[example.burn_in]
            desired = continuing_double_q_target(
                a["reward"][None],
                torch.tensor([a["done"]], device=self.device),
                a["q"][1:2],
                b["q"][1:2],
                self.cfg.gamma,
            )
            results.append((F.smooth_l1_loss(a["q"][0, row.action], desired[0], reduction="none"), a))
        return results

    def _her_examples(self, examples):
        if getattr(self.cfg, "controller_replay_state", "reconstruct") == "stored":
            from .controller_stored_replay import hindsight_examples

            return hindsight_examples(self, examples)
        core = self.actor_critic.core
        layout = HRLStateLayout(core.Hippo_n_feature)
        histories = []
        eligible_examples = []
        budgets = []
        for example in examples:
            row = example.rows[example.burn_in]
            budget = int(row.context[layout.countdown])
            if budget < 1:
                self.replay.reject("her_budget_expired")
                continue
            _, future = self.replay.sequence(row.key, 0, budget + 1)
            if len(future) < 2:
                self.replay.reject("her_no_future_achievement")
                continue
            prefix = list(example.rows[: example.burn_in])
            rows = prefix + future
            observations = {
                k: torch.as_tensor(np.stack([r.observation[k] for r in rows]), device=self.device)
                for k in row.observation
            }
            histories.append(
                ControllerHistory(
                    observations,
                    torch.tensor([r.index for r in rows], device=self.device),
                    torch.zeros(1, core.total_state_size, device=self.device),
                    torch.as_tensor(np.stack([r.condition for r in rows]), device=self.device),
                    rows[0].index == 0,
                    len(prefix),
                )
            )
            eligible_examples.append(example)
            budgets.append(budget)
        if not histories:
            return []
        operation = partial(reconstruct_controller_histories, decode=False)
        online = evaluate_replay(self.online_snapshot, operation, histories)
        target = evaluate_replay(self.target_snapshot, operation, histories)
        result = []
        for example, budget, a, b in zip(eligible_examples, budgets, online, target):
            dg = a["core_outputs"][:, : core.core_output_size].reshape(-1, core.Hippo_n_feature, core.expanded_length)[
                :, :, 0
            ]
            tdg = b["core_outputs"][:, : core.core_output_size].reshape(-1, core.Hippo_n_feature, core.expanded_length)[
                :, :, 0
            ]
            ids, active, count = current_dg_from_activity(dg)
            tids, tactive, tcount = current_dg_from_activity(tdg)
            eligible = [
                int(ids[i])
                for i in range(1, len(dg))
                if bool(active[i])
                and int(count[i]) == 1
                and dg[0, ids[i]] <= 0
                and bool(tactive[i])
                and int(tcount[i]) == 1
                and tids[i] == ids[i]
                and tdg[0, ids[i]] <= 0
            ]
            if not eligible:
                self.replay.reject("her_no_future_achievement")
                continue
            goal = eligible[int(self.her_rng.integers(len(eligible)))]
            result.append(replace(example, virtual_goal=goal, remaining=budget))
        return result

    def _controller_updates(self):
        main_seconds = her_seconds = 0.0
        # Under STOP, canonical recognition/events remain fixed between fresh
        # DG transactions and target refreshes. Q/worker updates cannot make a
        # rejected physical context compatible. JOINT invalidates every step.
        incompatible = set()
        for _ in range(self.clock.due(self.replay.accepted)):
            if self.cfg.ppo_dg_gradient != "stop":
                incompatible.clear()
            # Refresh online buffers after each optimizer transaction, target only
            # on its independent completed-main-update clock.
            self.online_snapshot.refresh(self.actor_critic, self.controller_version + 1)
            self.controller_version += 1
            examples = []
            started = time.perf_counter()
            keys = self.replay.candidate_order(incompatible)
            width = self._reconstruction_width()
            for start in range(0, len(keys), width):
                group = []
                group_keys = []
                for key in keys[start : start + width]:
                    try:
                        group.append(self._example(key))
                        group_keys.append(key)
                    except (ReplayRejected, ValueError) as exc:
                        if type(exc) is ValueError and str(exc) not in ("missing_history", "cross_episode"):
                            raise
                        self.replay.reject(str(exc))
                        incompatible.add(key)
                # Eligibility search must not retain autograd graphs for every
                # mostly-rejected candidate batch. Reconstruct the selected TD
                # positions under the same snapshot only after selection ends.
                with torch.no_grad():
                    results = self._evaluate_pairs(group)
                for key, example, result in zip(group_keys, group, results):
                    if isinstance(result, str):
                        self.replay.reject(result)
                        incompatible.add(key)
                        continue
                    examples.append(example)
                    if len(examples) == self.cfg.controller_td_positions:
                        break
                self._report_controller_progress()
                if len(examples) == self.cfg.controller_td_positions:
                    break
            main_seconds += time.perf_counter() - started
            if not examples:
                self.replay.reject("no_compatible_main_replay")
                break
            # If the full admissible population is smaller than the TD budget,
            # uniform resampling fills it without discarding valid main updates.
            available = len(examples)
            while len(examples) < self.cfg.controller_td_positions:
                examples.append(examples[int(self.replay.rng.integers(available))])
            started = time.perf_counter()
            selected_results = self._evaluate_pairs(examples)
            if any(isinstance(result, str) for result in selected_results):
                raise RuntimeError("Replay eligibility changed within the same optimizer snapshot")
            losses = [result[0] for result in selected_results]
            qs = [result[1]["q"].detach() for result in selected_results]
            main_loss = torch.stack(losses).mean()
            main_seconds += time.perf_counter() - started
            aux = []
            aux_qs = []
            started = time.perf_counter()
            if self.cfg.controller_her:
                # HER source material is independent of online command
                # authority: obsolete/inactive anchors remain valid exact
                # achieved endpoints, while structural DG generations do not.
                selected = []
                for key in self.replay.candidate_order():
                    try:
                        selected.append(self._example(key, allow_stale_anchor=True))
                    except (ReplayRejected, ValueError) as exc:
                        if type(exc) is ValueError and str(exc) not in ("missing_history", "cross_episode"):
                            raise
                        self.replay.reject(str(exc))
                    if len(selected) >= self.cfg.controller_her_positions:
                        break
                for result in self._evaluate_pairs(self._her_examples(selected)):
                    if isinstance(result, str):
                        self.replay.reject(result)
                    else:
                        aux.append(result[0])
                        aux_qs.append(result[1]["q"].detach())
            aux_loss = torch.stack(aux).mean() if aux else main_loss.new_zeros(())
            her_seconds += time.perf_counter() - started
            self.optimizer.zero_grad(set_to_none=True)
            started = time.perf_counter()
            main_loss.backward()
            main_seconds += time.perf_counter() - started
            if aux:
                started = time.perf_counter()
                (self.cfg.controller_her_loss_coeff * aux_loss).backward()
                her_seconds += time.perf_counter() - started
            if self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
            self._apply_lr(self.curr_lr)
            self.optimizer.step()
            self.controller_version += 1
            refresh = self.clock.finish(len(losses), len(aux))
            self._report_controller_progress()
            if refresh:
                self.target_snapshot.refresh(self.actor_critic, self.controller_version)
                incompatible.clear()
            all_q = torch.cat(qs)
            self.controller_stats.update(
                main_loss=float(main_loss.detach()),
                auxiliary_loss=float(aux_loss.detach()),
                q_mean=float(all_q.mean()),
                q_abs_max=float(all_q.abs().max()),
                auxiliary_q_mean=float(torch.cat(aux_qs).mean()) if aux_qs else 0.0,
                auxiliary_q_abs_max=float(torch.cat(aux_qs).abs().max()) if aux_qs else 0.0,
            )
        self.controller_stats["stored_state_replay"] = float(
            getattr(self.cfg, "controller_replay_state", "reconstruct") == "stored"
        )
        self.controller_stats.update(
            main_compute_seconds=main_seconds,
            her_compute_seconds=her_seconds,
            her_overhead_ratio=her_seconds / max(main_seconds, 1e-9),
        )

    def train(self, batch):
        started = time.perf_counter()
        self._ingest(batch)
        before = self.env_steps
        if self.cfg.controller_learning == "ddqn":
            with fresh_dg_parameter_owner(self.actor_critic):
                stats = super().train(batch)
        else:
            stats = super().train(batch)
        self.env_steps = before + (
            int(batch["controller_frames"].sum()) if self.cfg.summaries_use_frameskip else batch["actions"].numel()
        )
        self._controller_updates()
        with torch.no_grad(), self.param_server.policy_lock:
            self.actor_critic.controller_q.environment_decisions.fill_(self.replay.accepted)
            self.publication += 1
            self.actor_critic.controller_q.publication_version.fill_(self.publication)
            self.actor_critic.controller_q.fresh_version.fill_(self.train_step)
            self.published.load_state_dict(self.actor_critic.state_dict())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._published_versions[self.policy_id] = self.publication
        self._save_completed_frame_targets(before)
        stats = stats or {POLICY_ID_KEY: self.policy_id}
        stats[LEARNER_ENV_STEPS] = self.env_steps
        self.controller_stats.update(
            physical_interactions=self.replay.received,
            physical_frames=self.replay.physical_frames,
            accepted_replay_decisions=self.replay.accepted,
            main_updates=self.clock.completed,
            main_td_positions=self.clock.main_positions,
            auxiliary_td_positions=self.clock.auxiliary_positions,
            target_age=self.clock.completed - self.clock.target_at,
            update_debt=self.clock.due(self.replay.accepted),
            fresh_dg_steps=self.fresh_dg_steps,
            fresh_graph_batches=self.fresh_graph_batches,
            transaction_seconds=time.perf_counter() - started,
        )
        graph = self.actor_critic.core.policy_graph
        if graph is not None:
            self.controller_stats.setdefault("calibration_seconds", 0.0)
            ready = bool(graph.calibration_ready) if graph.contextual else False
            self.controller_stats.update(
                active_goal_count=float(graph.selectable_mask().sum()) if graph.contextual else 0.0,
                anchor_registrations=float(graph.anchor_registrations) if graph.contextual else 0.0,
                confirmation_attempts=float(graph.confirmation_attempts) if graph.contextual else 0.0,
                confirmation_successes=float(graph.confirmation_successes) if graph.contextual else 0.0,
                anchor_replacements=float(graph.anchor_replacements) if graph.contextual else 0.0,
                anchor_deactivations=float(graph.anchor_deactivations) if graph.contextual else 0.0,
                calibration_ready=float(ready),
                recognition_threshold=float(graph.recognition_threshold) if ready else 0.0,
                prediction_absolute_threshold=float(graph.prediction_absolute_threshold) if ready else 0.0,
                prediction_excess_threshold=float(graph.prediction_excess_threshold) if ready else 0.0,
                calibration_pair_count=float(graph.calibration_count) if graph.contextual else 0.0,
                activation_latency_mean=(
                    float(graph.activation_latency_sum / graph.activation_latency_count.clamp_min(1))
                    if graph.contextual
                    else 0.0
                ),
                empty_set_exploration=float(graph.empty_set_exploration_count),
                anchor_refinement_attempts=float(graph.anchor_refinement_attempts),
                anchor_refinements=float(graph.anchor_refinements),
                anchor_centrality_gain_mean=float(
                    graph.anchor_centrality_gain_sum / graph.anchor_refinements.clamp_min(1)
                ),
                anchor_age_mean=float(graph.anchor_age_sum / graph.anchor_age_count.clamp_min(1)),
                context_raw_multi_activation=float(graph.context_raw_multi_activation),
                context_accepted_events=float(graph.context_accepted_events),
                context_unique_rescues=float(graph.context_unique_rescues),
                context_zero_match=float(graph.context_zero_match),
                context_multi_match=float(graph.context_multi_match),
                positive_similarity_q10=float(graph.positive_similarity_q10),
                positive_similarity_q50=float(graph.positive_similarity_q50),
                positive_similarity_q90=float(graph.positive_similarity_q90),
                background_similarity_q50=float(graph.background_similarity_q50),
                background_similarity_q90=float(graph.background_similarity_q90),
                background_similarity_q99=float(graph.background_similarity_q99),
                background_above_threshold_fraction=float(graph.background_above_threshold_fraction),
                active_anchor_collision_fraction=float(graph.active_anchor_collision_fraction),
            )
            self.controller_stats.update(
                {f"command_slot_{slot:02d}": float(value) for slot, value in enumerate(graph.command_count)}
            )
        rejection_metrics = {
            "her_contextual_candidates": "her_contextual_candidate",
            "her_contextual_positive_hits": "her_contextual_positive_hit",
            "her_contextual_wrong_context": "her_contextual_same_dg_wrong_context",
            "her_contextual_start_achieved": "her_start_already_achieved_contextual",
            "her_contextual_missing_calibration": "her_contextual_missing_calibration",
            "her_terminal_successor_ca3_missing": "her_terminal_successor_ca3_missing",
        }
        self.controller_stats.update(
            {name: float(self.replay.rejected.get(reason, 0)) for name, reason in rejection_metrics.items()}
        )
        candidates = self.controller_stats["her_contextual_candidates"]
        self.controller_stats["her_contextual_positive_rate"] = (
            self.controller_stats["her_contextual_positive_hits"] / candidates if candidates else 0.0
        )
        for group, counts in self._optimizer_step_counts().items():
            self.controller_stats[group + "_optimizer_steps_min"] = min(counts.values(), default=0)
            self.controller_stats[group + "_optimizer_steps_max"] = max(counts.values(), default=0)
        self.controller_stats.update({f"rejected/{k}": v for k, v in self.replay.rejected.items()})
        summaries = stats.setdefault(TRAIN_STATS, {})
        modulation_stats = {}
        self._record_goal_modulation_summaries(modulation_stats)
        summaries.update({key: float(value) for key, value in modulation_stats.items()})
        summaries.update({f"controller/{k}": v for k, v in self.controller_stats.items()})
        return stats
