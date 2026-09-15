"""Goal-conditioned recurrent worker with a separately reconstructed target."""

import copy

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import advance_memory, double_dqn_target


class QWorker(nn.Module):
    schema = "intrmotiv/ddqn-worker/v2"

    def __init__(
        self,
        decoder,
        n_goals,
        bypass_size,
        repeat_width=8,
        length=64,
        write_modulation=None,
        intercept=2.43,
        n_actions=8,
        batch_independent=False,
    ):
        super().__init__()
        self.decoder = copy.deepcopy(decoder)
        self.batch_independent = bool(batch_independent)
        self.n_goals, self.bypass_size = n_goals, bypass_size
        self.repeat_width, self.width = repeat_width, repeat_width + length - 1
        self.intercept = float(intercept)
        self.write_modulation = (
            nn.Parameter(write_modulation.detach().clone()) if write_modulation is not None else None
        )
        hidden = decoder.get_out_size()
        self.joint = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.Tanh())
        self.q_head = nn.Linear(hidden, n_actions)

    def initial(self, batch, device=None):
        return torch.zeros(batch, self.n_goals, self.width, device=device)

    def activity(self, preactivation, goal):
        condition = F.one_hot(goal.long(), self.n_goals).to(preactivation.dtype)
        pre = preactivation.detach()
        if self.write_modulation is not None:
            scale, bias = (condition @ self.write_modulation).chunk(2, -1)
            pre = pre * (1 + scale) + bias
        return F.relu(pre - self.intercept)

    def step(self, memory, preactivation, bypass, goal, budget, episode_decision=None):
        memory = advance_memory(memory, self.activity(preactivation, goal), self.repeat_width)
        return self.readout(memory, bypass, goal, budget, episode_decision), memory

    def readout(self, memory, bypass, goal, budget, episode_decision=None):
        """Read already-updated memory. Shared by actors, TD learning and evaluation."""
        if bypass.shape != (len(goal), self.bypass_size):
            raise ValueError("parent depth/context bypass layout changed")
        condition = F.one_hot(goal.long(), self.n_goals).to(memory.dtype)
        state = torch.cat((memory.flatten(1), bypass.detach(), condition), -1)
        return self.readout_state(state, budget, episode_decision)

    def readout_state(self, state, budget, episode_decision=None):
        """Canonical evaluator core output includes memory, bypass and goal already."""
        if episode_decision is None:
            episode_decision = torch.zeros_like(budget)
        clocks = torch.stack((budget.float() / 64.0, episode_decision.float() / 1800.0), -1)
        return self.q_head(self.joint(torch.cat((self.decoder(state), clocks), -1)))

    def sequence(self, memory, pre, bypass, goals, budgets, clocks):
        """Batched decoder over physical time; preserve every CA3 transition.

        Only qualified row-independent decoders and goal-independent writes use
        this path. No replay sample, TD mask, update count or target changes.
        """
        if not self.batch_independent or self.write_modulation is not None:
            raise ValueError("batched execution requires a qualified row-independent frozen-write worker")
        if any(
            isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.Dropout)) and m.training for m in self.decoder.modules()
        ):
            raise ValueError("batch-dependent or stochastic decoder cannot use batched execution")
        memories = []
        for t in range(len(pre)):
            memory = advance_memory(memory, self.activity(pre[t], goals[t]), self.repeat_width)
            memories.append(memory)
        stacked = torch.stack(memories)
        t, b = goals.shape
        return self.readout(
            stacked.reshape(t * b, self.n_goals, self.width),
            bypass.reshape(t * b, self.bypass_size),
            goals.reshape(-1),
            budgets.reshape(-1),
            clocks.reshape(-1),
        ).reshape(t, b, -1)

    def rebuild(self, preactivation, bypass, goals, budgets, episode_start=False):
        if not episode_start and len(preactivation) < self.width:
            raise ValueError("missing washout prefix")
        memory = self.initial(preactivation.shape[1], preactivation.device)
        with torch.no_grad():
            for p, d, g, b in zip(preactivation, bypass, goals, budgets):
                memory = advance_memory(memory, self.activity(p, g), self.repeat_width)
        return memory


class DoubleDQNLearner:
    def __init__(self, worker, learning_rate=2e-4, target_period=1000, execution="reference"):
        if target_period < 1:
            raise ValueError("invalid target period")
        if execution not in ("reference", "batched"):
            raise ValueError("unknown learner execution backend")
        if execution == "batched" and (not worker.batch_independent or worker.write_modulation is not None):
            raise ValueError("batched execution requires a qualified row-independent frozen-write worker")
        self.execution = execution
        self.online = worker
        self.target = copy.deepcopy(worker).eval().requires_grad_(False)
        self.optimizer = torch.optim.Adam(worker.parameters(), lr=learning_rate)
        self.updates, self.target_period = 0, target_period

    def update(
        self,
        pre,
        bypass,
        goals,
        budgets,
        actions,
        rewards,
        done,
        mask,
        online_memory,
        target_memory,
        episode_decisions=None,
    ):
        """Input T+1 states already hold one virtual command and its budget."""
        online_q, target_q = [], []
        if episode_decisions is None:
            episode_decisions = torch.zeros_like(budgets)
        if self.execution == "batched":
            online_q = self.online.sequence(online_memory, pre, bypass, goals, budgets, episode_decisions)
            with torch.no_grad():
                target_q = self.target.sequence(target_memory, pre, bypass, goals, budgets, episode_decisions)
        else:
            for t in range(len(pre)):
                q, online_memory = self.online.step(
                    online_memory, pre[t], bypass[t], goals[t], budgets[t], episode_decisions[t]
                )
                online_q.append(q)
                with torch.no_grad():
                    q, target_memory = self.target.step(
                        target_memory, pre[t], bypass[t], goals[t], budgets[t], episode_decisions[t]
                    )
                    target_q.append(q)
            online_q, target_q = torch.stack(online_q), torch.stack(target_q)
        target = double_dqn_target(rewards, done, online_q[1:], target_q[1:])
        predicted = online_q[:-1].gather(-1, actions[..., None]).squeeze(-1)
        if not mask.any():
            raise ValueError("no valid loss positions")
        loss = F.smooth_l1_loss(predicted[mask], target[mask])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10, error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_period == 0:
            self.target.load_state_dict(self.online.state_dict(), strict=True)
        metrics = {
            "td_loss": float(loss.detach()),
            "grad_norm": float(norm),
            "valid_loss_positions": int(mask.sum()),
            "target_copies": self.updates // self.target_period,
            "td_target_min": float(target[mask].min()),
            "td_target_max": float(target[mask].max()),
            "td_target_mean": float(target[mask].mean()),
            "action_gap_mean": float(
                online_q[:-1].detach().topk(2, dim=-1).values.diff(dim=-1).neg().squeeze(-1)[mask].mean()
            ),
            "q_min": float(predicted[mask].detach().min()),
            "q_max": float(predicted[mask].detach().max()),
            "q_mean": float(predicted[mask].detach().mean()),
            "q_out_of_range_fraction": float(((predicted[mask] < 0) | (predicted[mask] > 1)).float().mean()),
        }

        for goal in goals[:-1][mask].unique().tolist():
            selected = mask & (goals[:-1] == goal)
            for name, values in (("prediction", predicted.detach()), ("td_target", target)):
                for stat in ("min", "max", "mean"):
                    metrics[f"goal_{goal}/{name}_{stat}"] = float(getattr(values[selected], stat)())
            gaps = online_q[:-1].detach().topk(2, dim=-1).values.diff(dim=-1).neg().squeeze(-1)
            metrics[f"goal_{goal}/action_gap_mean"] = float(gaps[selected].mean())
        return metrics
