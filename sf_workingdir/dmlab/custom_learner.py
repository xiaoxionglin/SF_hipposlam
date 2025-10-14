from __future__ import annotations


from typing import Dict, Optional, Tuple
import random

import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.env_info import EnvInfo
from sample_factory.algo.utils.model_sharing import ParameterServer
from sample_factory.algo.utils.tensor_dict import TensorDict, shallow_recursive_copy
from sample_factory.algo.utils.torch_utils import masked_select, synchronize, to_scalar
from sample_factory.algo.utils.rl_utils import gae_advantages
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import ActionDistribution, Config, PolicyID
from sample_factory.utils.utils import log
from sample_factory.utils.dicts import iterate_recursively


from sample_factory.algo.learning.learner import BaseLearner, DefaultLearner



class BaseDistanceRecorder(BaseLearner):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    def _record_distance_matrix(self, core_outputs, minibatch_size: int, masked_matrix: bool = True, return_progression: bool = False):
        locale_verbose = False
        if getattr(self.cfg, 'rec_distances', None) or getattr(self.cfg, 'distance_learning', None):
            R = getattr(self.cfg, 'Hippo_R', 8)
            L = getattr(self.cfg, 'Hippo_L', 48)
            hippo_n_feature = getattr(self.cfg, 'Hippo_n_feature', 64)
            # Total length of the shift register.
            expanded_length = R + L - 1
            # Core (shift register) output dimension.
            core_output_size = hippo_n_feature * expanded_length

            sequence_core = core_outputs[:, :core_output_size].view(minibatch_size, hippo_n_feature, expanded_length)

            progression = torch.argmax(torch.cat(((sequence_core != 0).to(dtype=torch.int), torch.ones(sequence_core.shape[:2] + (1,), dtype=torch.int)), dim=-1), dim=-1).squeeze(0)
            distance_matrix = torch.abs(progression.unsqueeze(-1) - progression.unsqueeze(-2)).to(dtype=torch.float)

            # sum = torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
            # value = sum/(distance_matrix.shape[1]**2)
            # meaned_value = value.mean().detach()
            if locale_verbose:
                log.info(f'RNN States shape: {core_outputs.shape}')
                log.info(f'SeququenceCore shape: {sequence_core.shape}')
                log.info(f'Progression: {progression}')
                log.info(f'Progression shape: {progression.shape}')
                log.info(f'Distance Matrix: {distance_matrix}')
                log.info(f'Distance Matrix shape: {distance_matrix.shape}')
                # log.info(f'Summed values: {value}')
                # log.info(f'Summed values shape: {value.shape}')

            if masked_matrix:
                masked_progression = torch.where(progression == sequence_core.shape[-1], False, True)
                distance_matrix_mask = torch.logical_and(masked_progression.unsqueeze(-1), masked_progression.unsqueeze(-2))
                masked_distance_matrix = distance_matrix * distance_matrix_mask
            else:
                masked_distance_matrix = None
        else:
            distance_matrix = None
        if return_progression:
            return distance_matrix, masked_distance_matrix, progression
        else:
            return distance_matrix, masked_distance_matrix
    
    
    def _train(
        self, gpu_buffer: TensorDict, batch_size: int, experience_size: int, num_invalids: int
    ) -> Optional[AttrDict]:
        timing = self.timing
        with torch.no_grad():
            early_stopping_tolerance = 1e-6
            early_stop = False
            prev_epoch_actor_loss = 1e9
            epoch_actor_losses = [0] * self.cfg.num_batches_per_epoch

            # recent mean KL-divergences per minibatch, this used by LR schedulers
            recent_kls = []

            if self.cfg.with_vtrace:
                assert (
                    self.cfg.recurrence == self.cfg.rollout and self.cfg.recurrence > 1
                ), "V-trace requires to recurrence and rollout to be equal"

            num_sgd_steps = 0
            stats_and_summaries: Optional[AttrDict] = None

            # When it is time to record train summaries, we randomly sample epoch/batch for which the summaries are
            # collected to get equal representation from different stages of training.
            # Half the time, we record summaries from the very large step of training. There we will have the highest
            # KL-divergence and ratio of PPO-clipped samples, which makes this data even more useful for analysis.
            # Something to consider: maybe we should have these last-batch metrics in a separate summaries category?
            with_summaries = self._should_save_summaries()
            if np.random.rand() < 0.5:
                summaries_epoch = np.random.randint(0, self.cfg.num_epochs)
                summaries_batch = np.random.randint(0, self.cfg.num_batches_per_epoch)
            else:
                summaries_epoch = self.cfg.num_epochs - 1
                summaries_batch = self.cfg.num_batches_per_epoch - 1

            assert self.actor_critic.training

        for epoch in range(self.cfg.num_epochs):
            with timing.add_time("epoch_init"):
                if early_stop:
                    break

                force_summaries = False
                minibatches = self._get_minibatches(batch_size, experience_size)

            for batch_num in range(len(minibatches)):
                with torch.no_grad(), timing.add_time("minibatch_init"):
                    indices = minibatches[batch_num]

                    # current minibatch consisting of short trajectory segments with length == recurrence
                    mb = self._get_minibatch(gpu_buffer, indices)

                    # enable syntactic sugar that allows us to access dict's keys as object attributes
                    mb = AttrDict(mb)

                with timing.add_time("calculate_losses"):
                    (
                        action_distribution,
                        policy_loss,
                        exploration_loss,
                        kl_old,
                        kl_loss,
                        value_loss,
                        loss_summaries,
                    ) = self._calculate_losses(mb, num_invalids)

                with timing.add_time("losses_postprocess"):
                    # noinspection PyTypeChecker
                    actor_loss: Tensor = policy_loss + exploration_loss + kl_loss
                    critic_loss = value_loss
                    loss: Tensor = actor_loss + critic_loss

                    epoch_actor_losses[batch_num] = float(actor_loss)

                    high_loss = 30.0
                    if torch.abs(loss) > high_loss:
                        log.warning(
                            "High loss value: l:%.4f pl:%.4f vl:%.4f exp_l:%.4f kl_l:%.4f (recommended to adjust the --reward_scale parameter)",
                            to_scalar(loss),
                            to_scalar(policy_loss),
                            to_scalar(value_loss),
                            to_scalar(exploration_loss),
                            to_scalar(kl_loss),
                        )

                        # perhaps something weird is happening, we definitely want summaries from this step
                        force_summaries = True

                with torch.no_grad(), timing.add_time("kl_divergence"):
                    # if kl_old is not None it is already calculated above
                    if kl_old is None:
                        # calculate KL-divergence with the behaviour policy action distribution
                        old_action_distribution = get_action_distribution(
                            self.actor_critic.action_space,
                            mb.action_logits,
                        )
                        kl_old = action_distribution.kl_divergence(old_action_distribution)
                        kl_old = masked_select(kl_old, mb.valids, num_invalids)

                    kl_old_mean = float(kl_old.mean().item())
                    recent_kls.append(kl_old_mean)
                    if kl_old.numel() > 0 and kl_old.max().item() > 100:
                        log.warning(f"KL-divergence is very high: {kl_old.max().item():.4f}")

                # update the weights
                with timing.add_time("update"):
                    # following advice from https://youtu.be/9mS1fIYj1So set grad to None instead of optimizer.zero_grad()
                    for p in self.actor_critic.parameters():
                        p.grad = None

                    loss.backward()
                    
                    if self.cfg.max_grad_norm > 0.0:
                        with timing.add_time("clip"):
                            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)

                    curr_policy_version = self.train_step  # policy version before the weight update

                    actual_lr = self.curr_lr
                    if num_invalids > 0:
                        # if we have masked (invalid) data we should reduce the learning rate accordingly
                        # this prevents a situation where most of the data in the minibatch is invalid
                        # and we end up doing SGD with super noisy gradients
                        actual_lr = self.curr_lr * (experience_size - num_invalids) / experience_size
                    self._apply_lr(actual_lr)

                    with self.param_server.policy_lock:
                        self.optimizer.step()

                    num_sgd_steps += 1

                with torch.no_grad(), timing.add_time("after_optimizer"):
                    self._after_optimizer_step()

                    if self.lr_scheduler.invoke_after_each_minibatch():
                        self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

                    # collect and report summaries
                    should_record_summaries = with_summaries
                    should_record_summaries &= epoch == summaries_epoch and batch_num == summaries_batch
                    should_record_summaries |= force_summaries
                    if should_record_summaries:
                        # hacky way to collect all of the intermediate variables for summaries
                        summary_vars = {**locals(), **loss_summaries}
                        stats_and_summaries = self._record_summaries(AttrDict(summary_vars))
                        del summary_vars
                        force_summaries = False

                    # make sure everything (such as policy weights) is committed to shared device memory
                    synchronize(self.cfg, self.device)
                    # this will force policy update on the inference worker (policy worker)
                    self.policy_versions_tensor[self.policy_id] = self.train_step

            # end of an epoch
            if self.lr_scheduler.invoke_after_each_epoch():
                self.curr_lr = self.lr_scheduler.update(self.curr_lr, recent_kls)

            new_epoch_actor_loss = float(np.mean(epoch_actor_losses))
            loss_delta_abs = abs(prev_epoch_actor_loss - new_epoch_actor_loss)
            if loss_delta_abs < early_stopping_tolerance:
                early_stop = True
                log.debug(
                    "Early stopping after %d epochs (%d sgd steps), loss delta %.7f",
                    epoch + 1,
                    num_sgd_steps,
                    loss_delta_abs,
                )
                break

            prev_epoch_actor_loss = new_epoch_actor_loss

        return stats_and_summaries
    
    def _record_summaries(self, train_loop_vars):
        var = train_loop_vars # TODO: Think of a better way, why is this necessary? Just redirecting pointer?
        stats = super()._record_summaries(train_loop_vars)
        if var.additional_stats["Distance Matrix"] != None:
            summed = torch.sum(torch.sum(var.additional_stats["Distance Matrix"].to(dtype=torch.float),dim=-1),dim=-1)
            value = summed/(var.additional_stats["Distance Matrix"].shape[1]**2)
            meaned_value, stded_value = torch.std_mean(value)
            stats.distance_metric = meaned_value.detach()
            stats.distance_metric_max = value.max().detach()
            stats.distance_metric_min = value.min().detach()
            stats.distance_metric_std = stded_value.detach()

            summed_masked = torch.sum(torch.sum(var.additional_stats["Distance Matrix Masked"].to(dtype=torch.float),dim=-1),dim=-1)
            value_masked = summed_masked/(var.additional_stats["Distance Matrix Masked"].shape[1]**2)
            meaned_value_masked, stded_value_masked = torch.std_mean(value_masked)
            stats.distance_metric_masked = meaned_value_masked.detach()
            stats.distance_metric_masked_max = value_masked.max().detach()
            stats.distance_metric_masked_min = value_masked.min().detach()
            stats.distance_metric_masked_std = stded_value_masked.detach()

            activated_sequences = var.additional_stats["Head Output"].count_nonzero(dim=-1).to(dtype=torch.float)
            meaned_activated_sequences, stded_activated_sequences = torch.std_mean(activated_sequences)
            stats.activated_sequences = meaned_activated_sequences.detach()
            stats.activated_sequences_max = activated_sequences.max().detach()
            stats.activated_sequences_min = activated_sequences.min().detach()
            stats.activated_sequences_std = stded_activated_sequences.detach()
        return stats


class DistanceLearnerSimple(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb = mb, 
            recurrence=recurrence, 
            valids = valids, 
            # grad_context=[True,True,True],
            return_outputs=[True,True,True])

        additional_stats["Head Output"] = outputs.head_outputs[:,:getattr(self.cfg, 'Hippo_n_feature', 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()
        

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(outputs.core_outputs, minibatch_size = outputs.minibatch_size, masked_matrix=True)
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            
            if self.cfg.masked_distance_matrix:
                adv = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
            else:
                adv = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')


        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)
            
            policy_loss += l1_loss
            
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            # value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            value_loss = torch.zeros(1)
        
        
        

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries

class DistanceLearnerEncoderDecoderSeparate(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
    
    @staticmethod
    def _make_grad_flip_hook(name):
        def _grad_flip_hook(module, grad_output: Tensor) -> Tensor:
            # log.debug(f"Flipping gradients on module {name}")
            return tuple(-g if g is not None else None for g in grad_output)
        return _grad_flip_hook
    
    def _register_backward_hooks(self):
        self.actor_critic.encoder.DG_projection.register_full_backward_pre_hook(DistanceLearnerEncoderDecoderSeparate._make_grad_flip_hook("encoder"))
        log.info("Succesfully registered backward hooks.")

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()

        # for param_group in self.optimizer.param_groups:
        # log.info(f'Parameter Group: {self.optimizer.param_groups[-1]}')

        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids
        
        outputs = self._forward_pass(
            mb = mb, 
            recurrence=recurrence, 
            valids = valids, 
            # grad_context=grad_context,
            return_outputs=[True,True,True])

        additional_stats["Head Output"] = outputs.head_outputs[:,:getattr(self.cfg, 'Hippo_n_feature', 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()
        

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(outputs.core_outputs, minibatch_size = outputs.minibatch_size, masked_matrix=True)
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            
            if self.cfg.masked_distance_matrix:
                adv = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
            else:
                adv = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
            
            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')


        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)
            
            policy_loss += l1_loss
            
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            # value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            value_loss = torch.zeros(1)
        
        
        

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries

class DistanceLearnerCombined(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
    
    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(mb = mb, recurrence=recurrence, valids = valids, return_outputs=[True,True,True])

        additional_stats["Head Output"] = outputs.head_outputs[:,:getattr(self.cfg, 'Hippo_n_feature', 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()
        

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(outputs.core_outputs, minibatch_size = outputs.minibatch_size, masked_matrix=True)
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio.cpu()
                values_cpu = values.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((outputs.num_trajectories * recurrence))
                adv = torch.zeros((outputs.num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns
            

            if self.cfg.masked_distance_matrix:
                advA = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
            else:
                advA = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            advA_std, advA_mean = torch.std_mean(masked_select(advA, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
                advA = (advA - advA_mean) / torch.clamp_min(advA_std, 1e-7)  # normalize advantage
            
            adv += advA
            

            # log.info(f'Advantage Shape: {adv.shape}')


        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)
            
            policy_loss += l1_loss
            
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries
    
class DistanceRecorder(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)
    
    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(mb = mb, recurrence=recurrence, valids = valids, return_outputs=[True,True,True])

        additional_stats["Head Output"] = outputs.head_outputs[:,:getattr(self.cfg, 'Hippo_n_feature', 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()
        

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(outputs.core_outputs, minibatch_size = outputs.minibatch_size, masked_matrix=True)
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            if self.cfg.with_vtrace:
                # V-trace parameters
                rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                c_hat = torch.Tensor([self.cfg.vtrace_c])

                ratios_cpu = ratio.cpu()
                values_cpu = values.cpu()
                rewards_cpu = mb.rewards_cpu
                dones_cpu = mb.dones_cpu

                vtrace_rho = torch.min(rho_hat, ratios_cpu)
                vtrace_c = torch.min(c_hat, ratios_cpu)

                vs = torch.zeros((outputs.num_trajectories * recurrence))
                adv = torch.zeros((outputs.num_trajectories * recurrence))

                next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                next_values /= self.cfg.gamma
                next_vs = next_values

                for i in reversed(range(self.cfg.recurrence)):
                    rewards = rewards_cpu[i::recurrence]
                    dones = dones_cpu[i::recurrence]
                    not_done = 1.0 - dones
                    not_done_gamma = not_done * self.cfg.gamma

                    curr_values = values_cpu[i::recurrence]
                    curr_vtrace_rho = vtrace_rho[i::recurrence]
                    curr_vtrace_c = vtrace_c[i::recurrence]

                    delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                    adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                    next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                    vs[i::recurrence] = next_vs

                    next_values = curr_values

                targets = vs.to(self.device)
                adv = adv.to(self.device)
            else:
                # using regular GAE
                adv = mb.advantages
                targets = mb.returns
            

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')


        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)
            
            policy_loss += l1_loss
            
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries





class DistanceLearnerMaster(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    @staticmethod
    def make_grad_flip_hook(name): # name useful for debugging only. This wrapper preserves the variable for the actual hook
        def grad_flip_hook(module, grad_output: Tensor) -> Tensor: # full_backward_pre_hook needs these inputs.
            # log.debug(f"Flipping gradients on module {name}")
            return tuple(-g if g is not None else None for g in grad_output)
        return grad_flip_hook
    
    def _register_backward_hooks(self):
        if self.cfg.encoder_decoder_share_losses:
            pass
        else:
            self.actor_critic.encoder.DG_projection.register_full_backward_pre_hook(DistanceLearnerMaster.make_grad_flip_hook("encoder.DG_projection"))
            log.info("Succesfully registered backward hooks.")

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb = mb, 
            recurrence=recurrence, 
            valids = valids, 
            # grad_context=[True,True,True],
            return_outputs=[True,True,True])

        additional_stats["Head Output"] = outputs.head_outputs[:,:getattr(self.cfg, 'Hippo_n_feature', 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()
        

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(outputs.core_outputs, minibatch_size = outputs.minibatch_size, masked_matrix=True)
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            
            if self.cfg.use_external:
                if self.cfg.with_vtrace:
                    # V-trace parameters
                    rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                    c_hat = torch.Tensor([self.cfg.vtrace_c])

                    ratios_cpu = ratio.cpu()
                    values_cpu = values.cpu()
                    rewards_cpu = mb.rewards_cpu
                    dones_cpu = mb.dones_cpu

                    vtrace_rho = torch.min(rho_hat, ratios_cpu)
                    vtrace_c = torch.min(c_hat, ratios_cpu)

                    vs = torch.zeros((outputs.num_trajectories * recurrence))
                    adv = torch.zeros((outputs.num_trajectories * recurrence))

                    next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                    next_values /= self.cfg.gamma
                    next_vs = next_values

                    for i in reversed(range(self.cfg.recurrence)):
                        rewards = rewards_cpu[i::recurrence]
                        dones = dones_cpu[i::recurrence]
                        not_done = 1.0 - dones
                        not_done_gamma = not_done * self.cfg.gamma

                        curr_values = values_cpu[i::recurrence]
                        curr_vtrace_rho = vtrace_rho[i::recurrence]
                        curr_vtrace_c = vtrace_c[i::recurrence]

                        delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                        adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                        next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                        vs[i::recurrence] = next_vs

                        next_values = curr_values

                    targets = vs.to(self.device)
                    adv = adv.to(self.device)
                else:
                    # using regular GAE
                    adv = mb.advantages
                    targets = mb.returns
            else:
                # Could this cause problems down the line?
                adv = torch.zeros(outputs.minibatch_size)
                # targets = torch.zeros(1)
            if self.cfg.use_internal:
                if self.cfg.metric == 'minimum':
                    metric = -torch.sum(torch.min(masked_distance_matrix.to(dtype=torch.float),dim=-1).values,dim=-1)
                elif self.cfg.metric == 'masked_sum':
                    metric = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
                elif self.cfg.metric == 'sum':
                    metric = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
                else:
                    raise NotImplementedError()
                adv += metric

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)
            
            policy_loss += l1_loss
            
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            if self.cfg.use_external:
                value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            else:
                value_loss = torch.zeros(1)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries
    


class DistanceLearnerReward(BaseDistanceRecorder):
    def __init__(
        self,
        cfg: Config,
        env_info: EnvInfo,
        policy_versions_tensor: Tensor,
        policy_id: PolicyID,
        param_server: ParameterServer,
    ):
        BaseLearner.__init__(self, cfg, env_info, policy_versions_tensor, policy_id, param_server)

    @staticmethod
    def make_grad_flip_hook(name): # name useful for debugging only. This wrapper preserves the variable for the actual hook
        def grad_flip_hook(module, grad_output: Tensor) -> Tensor: # full_backward_pre_hook needs these inputs.
            # log.debug(f"Flipping gradients on module {name}")
            return tuple(-g if g is not None else None for g in grad_output)
        return grad_flip_hook
    
    def _register_backward_hooks(self):
        if self.cfg.encoder_decoder_share_losses:
            pass
        else:
            self.actor_critic.encoder.DG_projection.register_full_backward_pre_hook(DistanceLearnerMaster.make_grad_flip_hook("encoder.DG_projection"))
            log.info("Succesfully registered backward hooks.")

    def _calculate_losses(
        self, mb: AttrDict, num_invalids: int
    ) -> Tuple[ActionDistribution, Tensor, Tensor | float, Optional[Tensor], Tensor | float, Tensor, Dict]:
        additional_stats = AttrDict()
        with torch.no_grad(), self.timing.add_time("losses_init"):
            recurrence: int = self.cfg.recurrence

            # PPO clipping
            clip_ratio_high = 1.0 + self.cfg.ppo_clip_ratio  # e.g. 1.1
            # this still works with e.g. clip_ratio = 2, while PPO's 1-r would give negative ratio
            clip_ratio_low = 1.0 / clip_ratio_high
            clip_value = self.cfg.ppo_clip_value

            valids = mb.valids

        outputs = self._forward_pass(
            mb = mb, 
            recurrence=recurrence, 
            valids = valids, 
            # grad_context=[True,True,True],
            return_outputs=[True,True,True])

        additional_stats["Head Output"] = outputs.head_outputs[:,:getattr(self.cfg, 'Hippo_n_feature', 64)]

        with self.timing.add_time("post_forward"):
            action_distribution = self.actor_critic.action_distribution()
            log_prob_actions = action_distribution.log_prob(mb.actions)
            ratio = torch.exp(log_prob_actions - mb.log_prob_actions)  # pi / pi_old

            # super large/small values can cause numerical problems and are probably noise anyway
            ratio = torch.clamp(ratio, 0.05, 20.0)

            values = outputs.result["values"].squeeze()
        

        # these computations are not the part of the computation graph
        with torch.no_grad(), self.timing.add_time("advantages_returns"):
            with self.timing.add_time("Distance Matrix"):
                distance_matrix, masked_distance_matrix = self._record_distance_matrix(outputs.core_outputs, minibatch_size = outputs.minibatch_size, masked_matrix=True)
                additional_stats["Distance Matrix"] = distance_matrix
                additional_stats["Distance Matrix Masked"] = masked_distance_matrix
            
            if self.cfg.use_external:
                if self.cfg.with_vtrace:
                    # V-trace parameters
                    rho_hat = torch.Tensor([self.cfg.vtrace_rho])
                    c_hat = torch.Tensor([self.cfg.vtrace_c])

                    ratios_cpu = ratio.cpu()
                    values_cpu = values.cpu()
                    rewards_cpu = mb.rewards_cpu #CHANGE HERE SUFFICIENT?
                    
                    dones_cpu = mb.dones_cpu

                    vtrace_rho = torch.min(rho_hat, ratios_cpu)
                    vtrace_c = torch.min(c_hat, ratios_cpu)

                    vs = torch.zeros((outputs.num_trajectories * recurrence))
                    adv = torch.zeros((outputs.num_trajectories * recurrence))

                    next_values = values_cpu[recurrence - 1 :: recurrence] - rewards_cpu[recurrence - 1 :: recurrence]
                    next_values /= self.cfg.gamma
                    next_vs = next_values

                    for i in reversed(range(self.cfg.recurrence)):
                        rewards = rewards_cpu[i::recurrence]
                        dones = dones_cpu[i::recurrence]
                        not_done = 1.0 - dones
                        not_done_gamma = not_done * self.cfg.gamma

                        curr_values = values_cpu[i::recurrence]
                        curr_vtrace_rho = vtrace_rho[i::recurrence]
                        curr_vtrace_c = vtrace_c[i::recurrence]

                        delta_s = curr_vtrace_rho * (rewards + not_done_gamma * next_values - curr_values)
                        adv[i::recurrence] = curr_vtrace_rho * (rewards + not_done_gamma * next_vs - curr_values)
                        next_vs = curr_values + delta_s + not_done_gamma * curr_vtrace_c * (next_vs - next_values)
                        vs[i::recurrence] = next_vs

                        next_values = curr_values

                    targets = vs.to(self.device)
                    adv = adv.to(self.device)
                else:
                    # using regular GAE
                    adv = mb.advantages
                    targets = mb.returns
            else:
                # Could this cause problems down the line?
                adv = torch.zeros(outputs.minibatch_size)
                # targets = torch.zeros(1)
            if self.cfg.use_internal:
                if self.cfg.metric == 'minimum':
                    metric = -torch.sum(torch.min(masked_distance_matrix.to(dtype=torch.float),dim=-1).values,dim=-1)
                elif self.cfg.metric == 'masked_sum':
                    metric = -torch.sum(torch.sum(masked_distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
                elif self.cfg.metric == 'sum':
                    metric = -torch.sum(torch.sum(distance_matrix.to(dtype=torch.float),dim=-1),dim=-1)
                else:
                    raise NotImplementedError()
                # log.info(f'Rewards Shape: {mb.rewards_cpu.shape}')
                # log.info(f'Metric Shape: {metric.shape}')
                # log.info(f'RNN states Shape: {mb.rnn_states.shape}')
                adv += metric

            adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))
            if self.cfg.normalize_advantage:
                adv = (adv - adv_mean) / torch.clamp_min(adv_std, 1e-7)  # normalize advantage
            # log.info(f'Advantage Shape: {adv.shape}')

        with self.timing.add_time("losses"):
            # noinspection PyTypeChecker
            policy_loss = self._policy_loss(ratio, adv, clip_ratio_low, clip_ratio_high, valids, num_invalids)
            l1_loss = self._l1_loss(outputs.head_outputs)
            
            policy_loss += l1_loss
            
            exploration_loss = self.exploration_loss_func(action_distribution, valids, num_invalids)
            kl_old, kl_loss = self.kl_loss_func(
                self.actor_critic.action_space, mb.action_logits, action_distribution, valids, num_invalids
            )
            old_values = mb["values"]
            if self.cfg.use_external:
                value_loss = self._value_loss(values, old_values, targets, clip_value, valids, num_invalids)
            else:
                value_loss = torch.zeros(1)

        loss_summaries = dict(
            ratio=ratio,
            clip_ratio_low=clip_ratio_low,
            clip_ratio_high=clip_ratio_high,
            values=outputs.result["values"],
            adv=adv,
            adv_std=adv_std,
            adv_mean=adv_mean,
            additional_stats=additional_stats,
        )
        del outputs

        return action_distribution, policy_loss, exploration_loss, kl_old, kl_loss, value_loss, loss_summaries
    
    def _prepare_batch(self, batch: TensorDict) -> Tuple[TensorDict, int, int]:
        with torch.no_grad():
            # create a shallow copy so we can modify the dictionary
            # we still reference the same buffers though
            buff = shallow_recursive_copy(batch)

            # ignore experience from other agents (i.e. on episode boundary) and from inactive agents
            valids: Tensor = buff["policy_id"] == self.policy_id
            # ignore experience that was older than the threshold even before training started
            curr_policy_version: int = self.train_step
            buff["valids"][:, :-1] = valids & (curr_policy_version - buff["policy_version"] < self.cfg.max_policy_lag)
            # for last T+1 step, we want to use the validity of the previous step
            buff["valids"][:, -1] = buff["valids"][:, -2]
            log.info(f'RNN_states Shape: {buff["rnn_states"].shape}')
            log.info(f'Internal Reward1: {buff["rewards"][:,:10]}')
            log.info(f'Reward Shape1: {buff["rewards"].shape}')
            # Calculate Internal Reward
            
            if True:#self.cfg.replace_reward:#False: # 
                buff["rewards_external"] = buff["rewards"].clone()
                rnn_state_shape = buff["rnn_states"].shape
                dataset_size = rnn_state_shape[0]*rnn_state_shape[1]
                distance_matrix, _, progression = self._record_distance_matrix(buff["rnn_states"].clone().reshape((dataset_size,) + tuple(rnn_state_shape[2:])), dataset_size, masked_matrix=False, return_progression=True)
                dataset_idx, row_idx = torch.where(progression == 0)
                # Only use first (row, col) pair for each row_idx
                lookup = {}
                for r, c in zip(dataset_idx.tolist(), row_idx.tolist()):
                    if r not in lookup:
                        lookup[r] = c
                # Prepare result tensor, filled with fallback value
                baseline = progression.shape[-1]
                internal_reward = torch.full((130, 1), baseline, dtype=torch.int32)

                # Fill in values where match was found
                # If all sequences got activated have a fallback_value
                fallback_value = 2*baseline
                for r, c in lookup.items():
                    vec = distance_matrix[r, c]
                    vec_mask = vec!=0
                    internal_reward[r] = vec[vec_mask].min() if vec_mask.numel() > 0 else fallback_value
                    # log.info(f'Adjusting internal reward at position {r,c} to be the minimum of {distance_matrix[r, c]}')
                buff["rewards"] = (-internal_reward.view(*rnn_state_shape[:2])[:, 1:]+baseline)*self.cfg.reward_scale
                log.info(f'Internal Reward2: {buff["rewards"][:,:10]}')
                log.info(f'Reward Shape2: {buff["rewards"].shape}')
                del lookup, rnn_state_shape, distance_matrix, progression, dataset_idx, row_idx, baseline, internal_reward, fallback_value, vec, vec_mask

            # ensure we're in train mode so that normalization statistics are updated
            if not self.actor_critic.training:
                self.actor_critic.train()

            buff["normalized_obs"] = self._prepare_and_normalize_obs(buff["obs"])
            del buff["obs"]  # don't need non-normalized obs anymore

            # calculate estimated value for the next step (T+1)
            normalized_last_obs = buff["normalized_obs"][:, -1]
            next_values = self.actor_critic(normalized_last_obs, buff["rnn_states"][:, -1], values_only=True)["values"]
            buff["values"][:, -1] = next_values

            if self.cfg.normalize_returns:
                # Since our value targets are normalized, the values will also have normalized statistics.
                # We need to denormalize them before using them for GAE caculation and value bootstrapping.
                # rl_games PPO uses a similar approach, see:
                # https://github.com/Denys88/rl_games/blob/7b5f9500ee65ae0832a7d8613b019c333ecd932c/rl_games/algos_torch/models.py#L51
                denormalized_values = buff["values"].clone()  # need to clone since normalizer is in-place
                self.actor_critic.returns_normalizer(denormalized_values, denormalize=True)
            else:
                # values are not normalized in this case, so we can use them as is
                denormalized_values = buff["values"]

            if self.cfg.value_bootstrap:
                # Value bootstrapping is a technique that reduces the surprise for the critic in case
                # we're ending the episode by timeout. Intuitively, in this case the cumulative return for the last step
                # should not be zero, but rather what the critic expects. This improves learning in many envs
                # because otherwise the critic cannot predict the abrupt change in rewards in a timed-out episode.
                # What we really want here is v(t+1) which we don't have because we don't have obs(t+1) (since
                # the episode ended). Using v(t) is an approximation that requires that rew(t) can be generally ignored.

                # Multiply by both time_out and done flags to make sure we count only timeouts in terminal states.
                # There was a bug in older versions of isaacgym where timeouts were reported for non-terminal states.
                buff["rewards"].add_(self.cfg.gamma * denormalized_values[:, :-1] * buff["time_outs"] * buff["dones"])

            if not self.cfg.with_vtrace:
                # calculate advantage estimate (in case of V-trace it is done separately for each minibatch)
                buff["advantages"] = gae_advantages(
                    buff["rewards"],
                    buff["dones"],
                    denormalized_values,
                    buff["valids"],
                    self.cfg.gamma,
                    self.cfg.gae_lambda,
                )
                # here returns are not normalized yet, so we should use denormalized values
                buff["returns"] = buff["advantages"] + buff["valids"][:, :-1] * denormalized_values[:, :-1]

            # remove next step obs, rnn_states, and values from the batch, we don't need them anymore
            for key in ["normalized_obs", "rnn_states", "values", "valids"]:
                buff[key] = buff[key][:, :-1]

            dataset_size = buff["actions"].shape[0] * buff["actions"].shape[1]
            for d, k, v in iterate_recursively(buff):
                # collapse first two dimensions (batch and time) into a single dimension
                d[k] = v.reshape((dataset_size,) + tuple(v.shape[2:]))

            buff["dones_cpu"] = buff["dones"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)
            buff["rewards_cpu"] = buff["rewards"].to("cpu", copy=True, dtype=torch.float, non_blocking=True)

            # return normalization parameters are only used on the learner, no need to lock the mutex
            if self.cfg.normalize_returns:
                self.actor_critic.returns_normalizer(buff["returns"])  # in-place

            num_invalids = dataset_size - buff["valids"].sum().item()
            if num_invalids > 0:
                invalid_fraction = num_invalids / dataset_size
                if invalid_fraction > 0.5:
                    log.warning(f"{self.policy_id=} batch has {invalid_fraction:.2%} of invalid samples")

                # invalid action values can cause problems when we calculate logprobs
                # here we set them to 0 just to be safe
                invalid_indices = (buff["valids"] == 0).nonzero().squeeze()
                buff["actions"][invalid_indices] = 0
                # likewise, some invalid values of log_prob_actions can cause NaNs or infs
                buff["log_prob_actions"][invalid_indices] = -1  # -1 seems like a safe value

            return buff, dataset_size, num_invalids



def make_hipposlam_learner(cfg: Config, env_info: EnvInfo, policy_versions_tensor: Tensor, policy_id: PolicyID, param_server: ParameterServer) -> BaseLearner:
    if cfg.distance_learning:
        return DistanceLearnerReward(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        # if not cfg.encoder_decoder_share_losses:
        #     if cfg.combined_learning:
        #         log.warn("Using encoder_decoder_share_losses & combined learning at the same time! Choosing DistanceLearnerEncoderDecoderSeparate.")
        #     return DistanceLearnerEncoderDecoderSeparate(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        # elif cfg.combined_learning:
        #     return DistanceLearnerCombined(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        # else:
        #     return DistanceLearnerSimple(cfg, env_info, policy_versions_tensor, policy_id, param_server)
    else:
        if cfg.rec_distances:
            return DistanceRecorder(cfg, env_info, policy_versions_tensor, policy_id, param_server)
        else:
            return DefaultLearner(cfg, env_info, policy_versions_tensor, policy_id, param_server)
