"""Batch independent replay histories without sharing online/target memories."""

import time

import torch


def prefix_memory(worker, prefix, device):
    if not prefix:
        return worker.initial(1, device)
    pre = torch.stack([r.observation.preactivation for r in prefix]).to(device)[:, None]
    bypass = torch.stack([r.observation.bypass for r in prefix]).to(device)[:, None]
    goals = torch.tensor([r.goal for r in prefix], device=device)[:, None]
    budgets = torch.tensor([r.budget for r in prefix], device=device)[:, None]
    return worker.rebuild(pre, bypass, goals, budgets, episode_start=prefix[0].index == 0)


def learn_batch(learner, samples, device):
    if any(s["relabeled"] for s in samples) and learner.online.write_modulation is not None:
        raise ValueError("write-conditioned virtual prefix reconstruction is not qualified")
    started = time.monotonic()
    length = max(len(s["segment"]) for s in samples)
    pre, bypass, goals, budgets, clocks, actions, rewards, dones, masks = [], [], [], [], [], [], [], [], []
    online_memory, target_memory = [], []
    batched = learner.execution == "batched"
    for s in samples:
        if not batched:
            online_memory.append(prefix_memory(learner.online, s["prefix"], device))
            target_memory.append(prefix_memory(learner.target, s["prefix"], device))
        rows, obs = s["segment"], s["observations"]
        t = len(rows)
        padded = obs + [obs[-1]] * (length - t)
        pre.append(torch.stack([o.preactivation for o in padded]))
        bypass.append(torch.stack([o.bypass for o in padded]))
        goals.append(torch.full((length + 1,), s["goal"], dtype=torch.long))
        budgets.append(torch.arange(s["budget"], s["budget"] - length - 1, -1).clamp_min(0))
        clocks.append(torch.arange(rows[0].index, rows[0].index + length + 1))
        actions.append(torch.tensor([r.action for r in rows] + [0] * (length - t)))
        rewards.append(torch.nn.functional.pad(s["reward"], (0, length - t)))
        dones.append(torch.nn.functional.pad(s["done"], (0, length - t), value=True))
        masks.append(torch.nn.functional.pad(s["mask"], (0, length - t), value=False))
    tensors = [torch.stack(x, 1).to(device) for x in (pre, bypass, goals, budgets, actions, rewards, dones, masks)]
    if batched:
        online_memory = batched_prefix_memory(learner.online, [s["prefix"] for s in samples], device)
        if prefix_signature(learner.online) == prefix_signature(learner.target):
            target_memory = online_memory  # read-only inputs; advance_memory never mutates
        else:
            target_memory = batched_prefix_memory(learner.target, [s["prefix"] for s in samples], device)
    else:
        online_memory, target_memory = torch.cat(online_memory), torch.cat(target_memory)
    prepared = time.monotonic()
    result = learner.update(*tensors, online_memory, target_memory, episode_decisions=torch.stack(clocks, 1).to(device))
    result["prefix_and_batch_seconds"] = prepared - started
    result["learner_update_seconds"] = time.monotonic() - prepared
    return result


def slice_attempt(sample, start, stop, width):
    """Slice losses without restarting the virtual deadline or physical memory."""
    result = dict(sample)
    result["prefix"] = (sample["prefix"] + sample["segment"][:start])[-width:]
    result["segment"] = sample["segment"][start:stop]
    result["observations"] = sample["observations"][start : stop + 1]
    result["budget"] = sample["budget"] - start
    for key in ("reward", "done", "mask"):
        result[key] = sample[key][start:stop]
    return result


class PositionBatcher:
    """Consume every sampled loss once, with an exact TD-position/update budget.

    A pending suffix carries its original deadline into the next update. No
    reward is dropped to make a batch fit; no extra HER gradient budget is hidden.
    """

    def __init__(self):
        self.pending = None
        self.backlog = []

    def sample(self, replay, registry, her_fraction, positions):
        samples, remaining = [], positions
        while remaining:
            if self.pending is None:
                try:
                    self.pending = self.backlog.pop(0) if self.backlog else replay.sample(registry, her_fraction)
                except ValueError:
                    self.backlog = samples + self.backlog
                    raise
            n = min(remaining, len(self.pending["segment"]))
            samples.append(slice_attempt(self.pending, 0, n, replay.width))
            self.pending = (
                slice_attempt(self.pending, n, len(self.pending["segment"]), replay.width)
                if n < len(self.pending["segment"])
                else None
            )
            remaining -= n
        return samples


def prefix_signature(worker):
    if worker.write_modulation is not None:
        raise ValueError("batched prefix reuse requires goal-independent writes")
    return (worker.n_goals, worker.width, worker.repeat_width, worker.intercept)


def batched_prefix_memory(worker, prefixes, device):
    """Right-align independent physical histories and reconstruct all at once.

    Padding injects zero activity *before* the real history; it cannot erase a
    short history or fabricate an observation. Different physical streams never
    share state. Frozen writes make online/target histories identical.
    """
    from .contracts import advance_memory

    prefix_signature(worker)
    width = max((len(p) for p in prefixes), default=0)
    memory = worker.initial(len(prefixes), device)
    if width == 0:
        return memory
    activity = []
    for prefix in prefixes:
        if prefix and prefix[0].index != 0 and len(prefix) < worker.width:
            raise ValueError("missing washout prefix")
        if any(not row.successor_valid for row in prefix):
            raise ValueError("invalid recurrent prefix")
        pre = torch.stack([r.observation.preactivation for r in prefix]) if prefix else torch.empty(0, worker.n_goals)
        values = torch.relu(pre.detach() - worker.intercept)
        activity.append(torch.nn.functional.pad(values, (0, 0, width - len(prefix), 0)))
    activity = torch.stack(activity, 1).to(device)
    with torch.no_grad():
        for row in activity:
            memory = advance_memory(memory, row, worker.repeat_width)
    return memory
