from __future__ import annotations

from typing import Dict, Optional

import gymnasium as gym
import torch
from torch import Tensor, nn

from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import (
    ActorCritic,
    ActorCriticSeparateWeights,
    ActorCriticSharedWeights,
    obs_space_without_action_mask,
)
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace
from sf_working_directories.IntrMotiv.dmlab.contextual_dg import DGTransitionPredictor


def controller_core_view(core_output: Tensor, ca3_size: int, ppo_dg_gradient: str = "stop") -> Tensor:
    """Apply the configured PPO gradient boundary at the CA3 prefix."""
    if ppo_dg_gradient == "joint":
        return core_output
    if ppo_dg_gradient != "stop":
        raise ValueError(f"Unknown ppo_dg_gradient={ppo_dg_gradient}")
    return torch.cat((core_output[:, : int(ca3_size)].detach(), core_output[:, int(ca3_size) :]), dim=-1)


class _PreserveMarkedInitializationMixin:
    """Keep explicitly marked pretrained submodules out of SF policy init."""

    def initialize_weights(self, layer):
        if getattr(layer, "_intrmotiv_preserve_initialization", False):
            return
        return super().initialize_weights(layer)


class TargetRelativeDecoder(nn.Module):
    """Goal decoder augmented with a target embedding and selected CA3 row."""

    def __init__(self, core, hidden_size: int = 128, embedding_size: int = 32):
        super().__init__()
        self.n_targets = core.Hippo_n_feature
        # Keep scalar layout metadata only. Registering the policy core as a child
        # of its decoder would create a cyclic module/state-dict hierarchy.
        self.core_output_size = core.core_output_size
        self.expanded_length = core.expanded_length
        self.target_condition_start = core.target_condition_start
        self.target_embedding = nn.Embedding(self.n_targets + 1, embedding_size)
        self.trace_projection = nn.Sequential(
            nn.Linear(core.expanded_length, embedding_size),
            nn.ReLU(),
        )
        self.network = nn.Sequential(
            nn.Linear(core.get_out_size() + 2 * embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder_out_size = hidden_size

    def forward(self, core_output: Tensor) -> Tensor:
        target_start = self.target_condition_start
        target = core_output[:, target_start : target_start + self.n_targets]
        valid = target.sum(dim=-1) > 0
        target_id = target.argmax(dim=-1)
        target_id = torch.where(valid, target_id, torch.full_like(target_id, self.n_targets))
        ca3 = core_output[:, : self.core_output_size].reshape(-1, self.n_targets, self.expanded_length)
        selected_trace = (ca3 * target.unsqueeze(-1)).sum(dim=1)
        relative = torch.cat((self.target_embedding(target_id), self.trace_projection(selected_trace)), dim=-1)
        return self.network(torch.cat((core_output, relative), dim=-1))

    def get_out_size(self) -> int:
        return self.decoder_out_size


class TargetFiLMDecoder(nn.Module):
    """Identity-initialized target-ID FiLM over a shared state decoder.

    The target one-hot selects modulation parameters only. It is deliberately
    removed from the state stream, and no target-specific CA3 trace is
    extracted, so the goal condition is invariant to how the target was
    previously encountered.
    """

    def __init__(self, core, hidden_size: int = 128):
        super().__init__()
        self.n_targets = core.Hippo_n_feature
        self.target_condition_start = core.target_condition_start
        state_input_size = core.get_out_size() - self.n_targets
        self.state_layer = nn.Sequential(
            nn.Linear(state_input_size, hidden_size),
            nn.ReLU(),
        )
        # A one-hot target left-multiplies this table to select one row. Keeping
        # this as an explicit parameter makes the conditioning operation
        # transparent and maps an all-zero no-target vector to zero modulation.
        self.target_modulation = nn.Parameter(torch.zeros(self.n_targets, 2 * hidden_size))
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder_out_size = hidden_size

    def _state_and_target(self, core_output: Tensor) -> tuple[Tensor, Tensor]:
        target_start = self.target_condition_start
        target_end = target_start + self.n_targets
        target = core_output[:, target_start:target_end]
        state = torch.cat((core_output[:, :target_start], core_output[:, target_end:]), dim=-1)
        return state, target

    def forward(self, core_output: Tensor) -> Tensor:
        state, target = self._state_and_target(core_output)
        hidden = self.state_layer(state)
        delta_scale, shift = (target @ self.target_modulation).chunk(2, dim=-1)
        modulated = hidden * (1.0 + delta_scale) + shift
        return self.output_layer(modulated)

    def get_out_size(self) -> int:
        return self.decoder_out_size


class ExplorationDecoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.decoder_out_size = hidden_size

    def forward(self, core_output: Tensor) -> Tensor:
        return self.network(core_output)

    def get_out_size(self) -> int:
        return self.decoder_out_size


class IntrMotivActorCriticSharedWeights(_PreserveMarkedInitializationMixin, ActorCriticSharedWeights):
    def __init__(self, model_factory, obs_space: ObsSpace, action_space: ActionSpace, cfg: Config):
        super().__init__(model_factory, obs_space, action_space, cfg)
        self.ppo_dg_gradient = getattr(cfg, "ppo_dg_gradient", "stop")
        if self.ppo_dg_gradient not in ("stop", "joint"):
            raise ValueError(f"Unknown ppo_dg_gradient={self.ppo_dg_gradient}")
        if self.ppo_dg_gradient == "joint" and bool(getattr(cfg, "iterative_update", False)):
            raise ValueError("ppo_dg_gradient=joint requires iterative_update=False")
        transition_prediction = getattr(cfg, "dg_transition_prediction", "none")
        self.dg_transition_predictor = None
        if transition_prediction != "none":
            self.dg_transition_predictor = DGTransitionPredictor(
                int(cfg.Hippo_n_feature),
                transition_prediction,
                int(getattr(cfg, "dg_transition_prediction_hidden_size", 128)),
            )
            self.dg_transition_predictor.apply(self.initialize_weights)
        goal_conditioning = getattr(cfg, "hrl_goal_conditioning", "legacy")
        if goal_conditioning == "target_trace":
            self.decoder = TargetRelativeDecoder(self.core)
        elif goal_conditioning == "target_id_film":
            self.decoder = TargetFiLMDecoder(self.core)

        if goal_conditioning in ("target_trace", "target_id_film"):
            decoder_size = self.decoder.get_out_size()
            self.critic_linear = nn.Linear(decoder_size, 1)
            self.action_parameterization = self.get_action_parameterization(decoder_size)
            self.decoder.apply(self.initialize_weights)
            self.critic_linear.apply(self.initialize_weights)
            self.action_parameterization.apply(self.initialize_weights)

        self.separate_goal_controllers = getattr(cfg, "intrinsic_goal_controller", "shared") == "separate"
        if getattr(self, "separate_goal_controllers", False):
            if getattr(cfg, "intrinsic_goal_mode", "none") == "none":
                raise ValueError("Separate goal controllers require intrinsic goals")
            n = int(cfg.Hippo_n_feature)
            self.goal_decoders = nn.ModuleList(
                model_factory.make_model_decoder_func(cfg, self.core.get_out_size()) for _ in range(n)
            )
            decoder_sizes = {decoder.get_out_size() for decoder in self.goal_decoders}
            if len(decoder_sizes) != 1:
                raise ValueError("Goal-specific decoder sizes differ")
            decoder_size = decoder_sizes.pop()
            self.goal_critics = nn.ModuleList(nn.Linear(decoder_size, 1) for _ in range(n))
            self.goal_action_parameterizations = nn.ModuleList(
                self.get_action_parameterization(decoder_size) for _ in range(n)
            )
            self.goal_decoders.apply(self.initialize_weights)
            self.goal_critics.apply(self.initialize_weights)
            self.goal_action_parameterizations.apply(self.initialize_weights)

        self.separate_exploration = getattr(cfg, "hrl_exploration_policy", "shared") == "separate"
        if self.separate_exploration:
            if not bool(getattr(cfg, "hrl_behavior_mode_condition", False)):
                raise ValueError("A separate exploration policy requires hrl_behavior_mode_condition=True")
            hidden = int(getattr(cfg, "hrl_exploration_decoder_size", 128))
            self.exploration_decoder = ExplorationDecoder(self.core.policy_base_output_size, hidden)
            self.exploration_critic_linear = nn.Linear(hidden, 1)
            self.exploration_action_parameterization = self.get_action_parameterization(hidden)
            self.exploration_decoder.apply(self.initialize_weights)
            self.exploration_critic_linear.apply(self.initialize_weights)
            self.exploration_action_parameterization.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = super().forward_head(normalized_obs_dict)
        if (
            getattr(self.cfg, "online_spatial_telemetry", False)
            and getattr(self.cfg, "dg_context_feedback", "none") == "none"
        ):
            self._online_behavior_dg = x[:, : int(self.cfg.Hippo_n_feature)].detach()
        return x

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        result = super().forward(normalized_obs_dict, rnn_states, values_only, action_mask)
        if getattr(self.cfg, "online_spatial_telemetry", False) and not values_only:
            contextual_activity = getattr(getattr(self, "core", None), "last_dg_activity", None)
            if torch.is_tensor(contextual_activity):
                self._online_behavior_dg = contextual_activity.detach()
            result["dg_activity"] = self._online_behavior_dg
        return result

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        # CA3 is a differentiable encoder view, but it is an explicit state
        # boundary for the controller. Only this prefix is stopped: visual
        # bypass, target, geometry, and manager-mode features retain their
        # existing PPO gradient paths.
        ca3_size = int(getattr(self.core, "core_output_size", 0))
        controller_output = controller_core_view(core_output, ca3_size, getattr(self, "ppo_dg_gradient", "stop"))
        if getattr(self, "separate_goal_controllers", False):
            start = self.core.target_condition_start
            target = controller_output[:, start : start + int(self.cfg.Hippo_n_feature)]
            valid = target.sum(-1, keepdim=True).gt(0)
            selector = torch.where(
                valid,
                target,
                torch.nn.functional.one_hot(
                    torch.zeros(target.size(0), dtype=torch.long, device=target.device), target.size(1)
                ).to(target.dtype),
            )
            decoded = [decoder(controller_output) for decoder in self.goal_decoders]
            values_by_goal = torch.stack(
                [critic(hidden).squeeze(-1) for critic, hidden in zip(self.goal_critics, decoded)], dim=1
            )
            values = (values_by_goal * selector).sum(dim=1)
            result = TensorDict(values=values)
            if values_only:
                return result
            params_by_goal = torch.stack(
                [head(hidden, action_mask)[0] for head, hidden in zip(self.goal_action_parameterizations, decoded)],
                dim=1,
            )
            action_params = (params_by_goal * selector.unsqueeze(-1)).sum(dim=1)
            self.last_action_distribution = get_action_distribution(self.action_space, action_params)
            result["action_logits"] = action_params
            self._maybe_sample_actions(sample_actions, result)
            return result
        if not self.separate_exploration:
            return super().forward_tail(controller_output, values_only, sample_actions, action_mask)

        goal_output = self.decoder(controller_output)
        exploration_output = self.exploration_decoder(controller_output[:, : self.core.policy_base_output_size])
        mode = controller_output[:, self.core.mode_condition_start : self.core.mode_condition_start + 5]
        free_exploration = mode[:, 2] > 0.5
        goal_values = self.critic_linear(goal_output).squeeze()
        exploration_values = self.exploration_critic_linear(exploration_output).squeeze()
        values = torch.where(free_exploration, exploration_values, goal_values)
        result = TensorDict(values=values)
        if values_only:
            return result

        goal_params, _ = self.action_parameterization(goal_output, action_mask)
        exploration_params, _ = self.exploration_action_parameterization(exploration_output, action_mask)
        action_params = torch.where(free_exploration.unsqueeze(-1), exploration_params, goal_params)
        self.last_action_distribution = get_action_distribution(self.action_space, action_params)
        result["action_logits"] = action_params
        result["free_exploration_mask"] = free_exploration
        self._maybe_sample_actions(sample_actions, result)
        return result


class IntrMotivActorCriticSeparateWeights(_PreserveMarkedInitializationMixin, ActorCriticSeparateWeights):
    pass


class ActorDoubleCriticSharedWeights(_PreserveMarkedInitializationMixin, ActorCritic):
    def __init__(
        self,
        model_factory,
        obs_space: ObsSpace,
        action_space: ActionSpace,
        cfg: Config,
    ):
        super().__init__(obs_space, action_space, cfg)

        # in case of shared weights we're using only a single encoder and a single core
        self.encoder = model_factory.make_model_encoder_func(cfg, obs_space)
        self.encoders = [self.encoder]  # a single shared encoder

        self.core = model_factory.make_model_core_func(cfg, self.encoder.get_out_size())

        self.decoder = model_factory.make_model_decoder_func(cfg, self.core.get_out_size())
        decoder_out_size: int = self.decoder.get_out_size()

        self.critic_linear_external = nn.Linear(decoder_out_size, 1)
        self.critic_linear_internal = nn.Linear(decoder_out_size, 1)
        self.action_parameterization = self.get_action_parameterization(decoder_out_size)

        self.apply(self.initialize_weights)

    def forward_head(self, normalized_obs_dict: Dict[str, Tensor]) -> Tensor:
        x = self.encoder(normalized_obs_dict)
        if getattr(self.cfg, "online_spatial_telemetry", False):
            self._online_behavior_dg = x[:, : int(self.cfg.Hippo_n_feature)].detach()
        return x

    def forward_core(self, head_output: Tensor, rnn_states):
        x, new_rnn_states = self.core(head_output, rnn_states)
        return x, new_rnn_states

    def forward_tail(
        self, core_output, values_only: bool, sample_actions: bool, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        decoder_output = self.decoder(core_output)
        values_external = self.critic_linear_external(decoder_output).squeeze()
        values_internal = self.critic_linear_internal(decoder_output).squeeze()

        result = TensorDict(values_external=values_external, values_internal=values_internal)
        if values_only:
            return result

        action_distribution_params, self.last_action_distribution = self.action_parameterization(
            decoder_output, action_mask
        )

        # `action_logits` is not the best name here, better would be "action distribution parameters"
        result["action_logits"] = action_distribution_params

        self._maybe_sample_actions(sample_actions, result)
        return result

    def forward(
        self, normalized_obs_dict, rnn_states, values_only=False, action_mask: Optional[Tensor] = None
    ) -> TensorDict:
        x = self.forward_head(normalized_obs_dict)
        x, new_rnn_states = self.forward_core(x, rnn_states)
        result = self.forward_tail(x, values_only, sample_actions=True, action_mask=action_mask)
        result["new_rnn_states"] = new_rnn_states
        if getattr(self.cfg, "online_spatial_telemetry", False) and not values_only:
            result["dg_activity"] = self._online_behavior_dg
        return result


def make_hipposlam_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.model_context import global_model_factory

    model_factory = global_model_factory()
    obs_space = obs_space_without_action_mask(obs_space)
    online_spatial = bool(getattr(cfg, "online_spatial_telemetry", False))
    if online_spatial:
        if not isinstance(obs_space, gym.spaces.Dict) or "telemetry_pose" not in obs_space.spaces:
            raise ValueError("online spatial telemetry requires a declared telemetry_pose observation")
        filtered_spaces = obs_space.spaces.copy()
        del filtered_spaces["telemetry_pose"]
        obs_space = gym.spaces.Dict(filtered_spaces)

    uses_new_worker = (
        getattr(cfg, "hrl_goal_conditioning", "legacy") in ("target_trace", "target_id_film")
        or getattr(cfg, "hrl_exploration_policy", "shared") == "separate"
    )
    if uses_new_worker and not cfg.actor_critic_share_weights:
        raise ValueError("Custom goal and separate-exploration workers require shared actor-critic weights")
    if uses_new_worker and bool(getattr(cfg, "double_value", False)):
        raise ValueError("Custom goal and separate-exploration workers do not support double_value")

    if online_spatial and not cfg.actor_critic_share_weights:
        raise ValueError("online spatial telemetry currently requires shared actor-critic weights")

    if cfg.actor_critic_share_weights:
        if cfg.distance_learning:
            if cfg.double_value:
                actor_critic = ActorDoubleCriticSharedWeights(model_factory, obs_space, action_space, cfg)
            else:
                actor_critic = IntrMotivActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
        else:
            actor_critic = IntrMotivActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
    else:
        actor_critic = IntrMotivActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)
    actor_critic.privileged_obs_keys = ("telemetry_pose",) if online_spatial else ()
    return actor_critic
