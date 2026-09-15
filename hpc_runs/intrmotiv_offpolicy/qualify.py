"""Deterministic production-readout qualification; no DMLab or tabular substitute."""

import argparse
import json

import torch
from torch import nn

from .worker import DoubleDQNLearner, QWorker


class FixtureDecoder(nn.Module):
    def __init__(self, inputs, hidden=64):
        super().__init__()
        self.layer = nn.Linear(inputs, hidden)

    def forward(self, x):
        return torch.tanh(self.layer(x))

    def get_out_size(self):
        return self.layer.out_features


def expressivity(updates=6000):
    torch.manual_seed(91)
    worker = QWorker(FixtureDecoder(4), 1, 2, repeat_width=1, length=1, n_actions=2)
    # At near state action 0 hits immediately. At far state action 1 takes
    # two steps; action 0 takes three. Budget 1: both fail; budget 2: choose 1.
    # Third state: action 0 needs 2 steps, action 1 needs 1 => switch versus far.
    bypass = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, -1.0]]
    )
    budgets = torch.tensor([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    expected = torch.tensor(
        [[1.0, 0.0], [1.0, 0.99], [0.0, 0.0], [0.0, 0.99], [0.0, 1.0], [0.99, 1.0], [0.5, 0.0], [0.5, 0.99]]
    )
    opt = torch.optim.Adam(worker.parameters(), lr=0.003)
    schedule = torch.optim.lr_scheduler.StepLR(opt, step_size=4000, gamma=0.1)
    for _ in range(updates):
        q = worker.readout(worker.initial(8), bypass, torch.zeros(8, dtype=torch.long), budgets)
        loss = (q - expected).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        schedule.step()
    q = worker.readout(worker.initial(8), bypass, torch.zeros(8, dtype=torch.long), budgets)
    error = float((q - expected).abs().max().detach())
    # Last state: action 0 has immediate success probability 0.5, otherwise
    # absorbing failure; action 1 succeeds deterministically in two steps.
    # This known finite-horizon task strictly switches optimal action with budget.
    action_switch = q[-2:].argmax(-1).tolist() == [0, 1]
    return dict(
        max_abs_error=error,
        q=q.detach().tolist(),
        expected=expected.tolist(),
        action_switch=action_switch,
        passed=error < 0.04 and action_switch,
    )


def chain(length, period, updates=10000):
    torch.manual_seed(123)
    worker = QWorker(FixtureDecoder(length + 3), 1, length + 1, repeat_width=1, length=1, n_actions=2)
    learner = DoubleDQNLearner(worker, learning_rate=0.002, target_period=period)
    states = torch.arange(length).repeat_interleave(2)
    actions = torch.tensor([0, 1] * length)[None]
    pre = torch.zeros(2, 2 * length, 1)
    bypass = torch.stack(
        (torch.nn.functional.one_hot(states, length + 1), torch.nn.functional.one_hot(states + 1, length + 1))
    ).float()
    budgets = torch.stack((length - states, length - states - 1))
    goals = torch.zeros(2, 2 * length, dtype=torch.long)
    reward = ((states == length - 1) & (actions[0] == 0)).float()[None]
    done = ((states == length - 1) | (actions[0] == 1))[None]
    mask = torch.ones_like(done)
    for _ in range(updates):
        learner.update(
            pre,
            bypass,
            goals,
            budgets,
            actions,
            reward,
            done,
            mask,
            worker.initial(2 * length),
            worker.initial(2 * length),
        )
    with torch.no_grad():
        q, _ = worker.step(
            worker.initial(length),
            torch.zeros(length, 1),
            torch.eye(length + 1)[:length],
            torch.zeros(length, dtype=torch.long),
            torch.arange(length, 0, -1),
        )
    expected = 0.99 ** torch.arange(length - 1, -1, -1)
    error = float((q[:, 0] - expected).abs().max())
    return dict(
        length=length,
        target_period_updates=period,
        optimizer_updates=updates,
        valid_td_positions=updates * 2 * length,
        unique_environment_transitions=2 * length,
        target_copies=updates // period,
        start_q=float(q[0, 0]),
        expected_start=float(expected[0]),
        max_abs_error=error,
        greedy_success=bool((q[:, 0] > q[:, 1]).all()),
        passed=error < 0.08 and bool((q[:, 0] > q[:, 1]).all()),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    results = dict(
        scope="Synthetic deterministic replay; not DMLab performance",
        expressivity=expressivity(),
        chains=[chain(n, p) for n in (4, 16, 40, 64) for p in (1000, 100)],
    )
    results["selected_target_period_updates"] = 100
    results["passed"] = results["expressivity"]["passed"] and all(
        r["passed"] for r in results["chains"] if r["target_period_updates"] == 100
    )
    from pathlib import Path

    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0 if results["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
