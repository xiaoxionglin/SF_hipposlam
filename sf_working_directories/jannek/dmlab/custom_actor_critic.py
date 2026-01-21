from __future__ import annotations

from typing import Dict, Optional

from torch import Tensor, nn
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import (
    ActorCritic, 
    ActorCriticSharedWeights, 
    ActorCriticSeparateWeights, 
    obs_space_without_action_mask
)
from sample_factory.utils.typing import ActionSpace, Config, ObsSpace


class ActorDoubleCriticSharedWeights(ActorCritic):
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
        return result
    

def make_hipposlam_actor_critic(cfg: Config, obs_space: ObsSpace, action_space: ActionSpace) -> ActorCritic:
    from sample_factory.algo.utils.model_context import global_model_factory

    model_factory = global_model_factory()
    obs_space = obs_space_without_action_mask(obs_space)

    if cfg.actor_critic_share_weights:
        if cfg.distance_learning:
            if cfg.double_value:
                return ActorDoubleCriticSharedWeights(model_factory, obs_space, action_space, cfg)
            else:
                return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
        else:
            return ActorCriticSharedWeights(model_factory, obs_space, action_space, cfg)
    else:
        return ActorCriticSeparateWeights(model_factory, obs_space, action_space, cfg)
