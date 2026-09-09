"""Default-off learner extension: PPO is unchanged; a shadow model learns beside it."""

import gymnasium as gym
import torch

from sample_factory.algo.learning.learner import DefaultLearner, default_make_learner_func
from sample_factory.utils.utils import log, str2bool
from sf_xxl.dmlab.world_model import DGEventWorldModel, event_prediction_loss, next_dg_event_labels


def add_world_model_args(parser):
    parser.add_argument("--dg_world_model", type=str2bool, default=False,
                        help="Train an action-conditioned next-DG-event shadow model; does not control PPO.")
    parser.add_argument("--dg_world_model_horizon", type=int, default=16, help="Prediction horizon in policy decisions.")
    parser.add_argument("--dg_world_model_hidden_size", type=int, default=128)
    parser.add_argument("--dg_world_model_lr", type=float, default=0.001)


def make_world_model_learner(cfg, env_info, policy_versions_tensor, policy_id, param_server):
    if not getattr(cfg, "dg_world_model", False):
        return default_make_learner_func(cfg, env_info, policy_versions_tensor, policy_id, param_server)
    return DGWorldModelLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)


class DGWorldModelLearner(DefaultLearner):
    def __init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server):
        if not isinstance(env_info.action_space, gym.spaces.Discrete):
            raise ValueError("dg_world_model currently supports a single Discrete action space")
        if not cfg.use_rnn or not cfg.actor_critic_share_weights or cfg.recurrence < 2:
            raise ValueError("dg_world_model requires a shared recurrent actor/critic and recurrence >= 2")
        if cfg.core_name not in ("BypassSS", "BypassFixedRNN", "simple_sequence", "fixed_rnn"):
            raise ValueError("dg_world_model requires a continuous DG shift-register core (e.g. BypassSS)")
        if min(cfg.dg_world_model_horizon, cfg.dg_world_model_hidden_size, cfg.dg_world_model_lr) <= 0:
            raise ValueError("World-model horizon, hidden size and learning rate must be positive")
        super().__init__(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        self.world_model = self.world_model_optimizer = None
        self._world_model_checkpoint = self._world_model_pending = None
        self._world_model_stats = {}

    def init(self):
        result = super().init()
        self._initialize_world_model()
        if self._world_model_checkpoint is not None:
            self._restore_world_model(self._world_model_checkpoint)
            self._world_model_checkpoint = None
        if self.cfg.dg_world_model_horizon >= self.cfg.recurrence:
            log.warning("DG prediction horizon >= recurrence: negatives are censored; inspect usable_fraction.")
        return result

    def _world_model_config(self):
        return dict(ca3_size=self.cfg.Hippo_n_feature * (self.cfg.Hippo_R + self.cfg.Hippo_L - 1),
                    n_features=self.cfg.Hippo_n_feature, action_count=self.env_info.action_space.n,
                    hidden_size=self.cfg.dg_world_model_hidden_size, horizon=self.cfg.dg_world_model_horizon)

    def _initialize_world_model(self):
        config = self._world_model_config()
        # Do not perturb the actor's sampling RNG merely by enabling the head.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device="cpu").manual_seed((self.cfg.seed or 0) + 1701 + self.policy_id)
            torch.set_rng_state(generator.get_state())
            self.world_model = DGEventWorldModel(**{k: v for k, v in config.items() if k != "horizon"})
        self.world_model.to(self.device)
        self.world_model_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=self.cfg.dg_world_model_lr)

    def _forward_pass(self, mb, recurrence, valids, return_outputs=(True, True, True), head_only=False):
        requested = list(return_outputs)
        if not head_only:
            requested[0] = requested[1] = True
        outputs = super()._forward_pass(mb, recurrence, valids, requested, head_only)
        if not head_only:
            n, f = self._world_model_config()["ca3_size"], self.cfg.Hippo_n_feature
            if outputs.core_outputs.size(-1) < n or outputs.head_outputs.size(-1) < f:
                raise ValueError("World model cannot locate the configured CA3/DG prefix")
            # Uses the SAME forward pass as PPO, never a second encoder pass.
            # build_core_out_from_seq restored the chronological segment order.
            self._world_model_pending = (
                outputs.core_outputs[:, :n].detach(), outputs.head_outputs[:, :f].detach(),
                mb.actions.detach(), mb.dones.detach(), valids.detach(), recurrence,
            )
        return outputs

    def _after_optimizer_step(self):
        super()._after_optimizer_step()
        if self._world_model_pending is None:
            return
        ca3, dg, actions, dones, valids, recurrence = self._world_model_pending
        self._world_model_pending = None
        horizon = self.cfg.dg_world_model_horizon
        labels = next_dg_event_labels(dg.reshape(-1, recurrence, dg.size(-1)),
                                     dones.reshape(-1, recurrence), valids.reshape(-1, recurrence), horizon)
        # Invalid/padded samples may use an action sentinel. They supply no
        # supervision and must not be passed as candidate actions to the head.
        actions = torch.where(valids.reshape(-1), actions.reshape(-1), 0)
        # The SF hook is invoked inside no_grad; explicitly enable this isolated update.
        with torch.enable_grad():
            loss, stats = event_prediction_loss(self.world_model, ca3, actions, labels, horizon)
            self.world_model_optimizer.zero_grad(set_to_none=True)
            if labels.usable.any():
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite DG world-model loss")
                loss.backward()
                self.world_model_optimizer.step()
        stats["usable_fraction"] = labels.usable.sum().float() / valids.sum().clamp_min(1)
        self._world_model_stats = {"dg_world_model/" + key: value.item() for key, value in stats.items()}

    def _record_summaries(self, train_loop_vars):
        stats = super()._record_summaries(train_loop_vars)
        stats.update(self._world_model_stats)
        return stats

    def _get_checkpoint_dict(self):
        checkpoint = super()._get_checkpoint_dict()
        if self.world_model is not None:
            checkpoint["dg_world_model"] = dict(schema=1, config=self._world_model_config(),
                                               model=self.world_model.state_dict(),
                                               optimizer=self.world_model_optimizer.state_dict())
        return checkpoint

    def _restore_world_model(self, state):
        if state.get("schema") != 1 or state["config"] != self._world_model_config():
            raise ValueError("DG world-model checkpoint config mismatch; use a new experiment for a changed head")
        self.world_model.load_state_dict(state["model"])
        self.world_model_optimizer.load_state_dict(state["optimizer"])

    def _load_state(self, checkpoint_dict, load_progress=True):
        super()._load_state(checkpoint_dict, load_progress)
        state = checkpoint_dict.get("dg_world_model")
        if self.world_model is None:
            self._world_model_checkpoint = state
        elif state is None:
            self._initialize_world_model()
        else:
            self._restore_world_model(state)
