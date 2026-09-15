"""SF model factory: frozen source features, CA3 state and epsilon-greedy Q."""

from dataclasses import dataclass

import torch
from torch import nn

from .checkpoint import convert_actor
from .features import FrozenParentFeatures


@dataclass
class NativeModelFactory:
    parent_factory: object
    parent_cfg: object
    checkpoint: str
    action_vectors: object
    counter: object
    counter_lock: object

    def __call__(self, cfg, obs_space, action_space):
        torch.manual_seed(cfg.seed)
        from gymnasium import spaces

        from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict

        parent_space = spaces.Dict({k: v for k, v in obs_space.spaces.items() if k != "ddqn_identity"})
        parent = self.parent_factory(self.parent_cfg, parent_space, action_space)
        parent.model_to_device(torch.device("cpu"))
        parent.load_state_dict(load_checkpoint_dict(self.checkpoint, torch.device("cpu"))["model"], strict=True)
        worker, report = convert_actor(parent, self.parent_cfg, self.action_vectors, cfg.seed)
        return NativeActor(
            parent, worker, cfg.ddqn_registry, self.counter, self.counter_lock, report, cfg.ddqn_inference_threads
        )


class NativeActor(nn.Module):
    def __init__(self, parent, worker, registry, counter, counter_lock, report, inference_threads=1):
        super().__init__()
        self.parent = parent
        self.worker = worker
        self.extractor = FrozenParentFeatures(parent, exclusive=parent.core.topological_enabled)
        self.registry = list(registry)
        self.counter, self.counter_lock = counter, counter_lock
        self.report = report
        self.device = torch.device("cpu")
        self.inference_threads = inference_threads

    def model_to_device(self, device):
        torch.set_num_threads(self.inference_threads)
        self.device = torch.device(device)
        self.to(self.device)
        self.parent.model_to_device(self.device)

    def device_for_input_tensor(self, key):
        return self.device

    def type_for_input_tensor(self, key):
        return torch.int64 if key == "ddqn_identity" else self.parent.type_for_input_tensor(key)

    def normalize_obs(self, obs):
        # FrozenParentFeatures performs source normalization exactly once.
        return obs

    def train(self, mode=True):
        super().train(mode)
        self.parent.eval()
        return self

    def forward(self, obs, rnn_states, action_mask=None):
        from sample_factory.algo.utils.tensor_dict import TensorDict

        if action_mask is not None:
            raise ValueError("navigation8 adapter does not support action masks")
        rows = self.extractor(obs)
        pre = torch.stack([r.preactivation for r in rows])
        bypass = torch.stack([r.bypass for r in rows])
        events = torch.stack([r.events for r in rows])
        count = len(rows)
        width = self.worker.n_goals * self.worker.width
        memory = rnn_states[:, :width].reshape(count, self.worker.n_goals, self.worker.width)
        goals = rnn_states[:, width].long()
        budgets = rnn_states[:, width + 1].long()
        options = rnn_states[:, width + 2].long()
        hit = events.gather(1, goals[:, None]).squeeze(1)
        choose = (budgets <= 0) | hit
        if choose.any():
            registry = torch.tensor(self.registry, device=pre.device)
            allowed = (~events[choose][:, registry]).float()
            if (allowed.sum(1) == 0).any():
                raise ValueError("no nontrivial command available")
            goals = goals.clone()
            budgets = budgets.clone()
            options = options.clone()
            goals[choose] = registry[torch.multinomial(allowed, 1).squeeze(1)]
            budgets[choose] = 64
            options[choose] += 1
        q, memory = self.worker.step(memory, pre, bypass, goals, budgets, obs["ddqn_identity"][:, 2])
        # A separate shared counter is not overwritten by parameter publication.
        with self.counter_lock:
            first = int(self.counter.item())
            self.counter.add_(count)
        eps = (1 - 0.9 * (first + torch.arange(count, device=pre.device)) / 250000).clamp(min=0.1)
        greedy = q.argmax(-1)
        explore = torch.rand(count, device=pre.device) < eps
        actions = torch.where(explore, torch.randint(0, q.shape[-1], (count,), device=pre.device), greedy)
        probability = eps / q.shape[-1] + (actions == greedy) * (1 - eps)
        packet = torch.cat((pre, bypass, events.float(), goals[:, None], budgets[:, None], options[:, None]), 1)
        return TensorDict(
            actions=actions,
            action_logits=q,
            log_prob_actions=probability.log(),
            values=q.max(-1).values,
            ddqn_packet=packet,
            new_rnn_states=torch.cat((memory.flatten(1), goals[:, None], (budgets - 1)[:, None], options[:, None]), 1),
        )
