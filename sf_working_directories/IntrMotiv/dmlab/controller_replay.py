"""Owned physical replay, without frozen-DG representations or virtual graph rows."""

import copy
from collections import OrderedDict
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from hpc_runs.intrmotiv_offpolicy.sf_transport import OrderedIngress


@dataclass(frozen=True)
class PhysicalDecision:
    stream: tuple
    episode: int
    index: int
    serial: int
    observation: dict
    action: int
    condition: np.ndarray
    context: np.ndarray
    publication: int
    generation: int
    terminated: bool
    truncated: bool
    physical_frames: int
    successor: dict | None = None
    successor_valid: bool = False
    worker_state: np.ndarray | None = None
    terminal_dg: np.ndarray | None = None
    terminal_publication: int | None = None
    real_reward: float | None = None
    real_events: dict | None = None

    @property
    def key(self):
        return self.stream, self.episode, self.index


class PhysicalReplay:
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.rows = OrderedDict()
        self.rng = np.random.default_rng(seed)
        self.ingress = OrderedIngress(max_pending=capacity)
        self.received = self.accepted = self.physical_frames = 0
        self.session = 0
        self.rejected = {}

    def receive(self, row):
        # Ingress validation precedes accounting: duplicates are not interactions.
        owned = copy.deepcopy(row)
        self.ingress.add(row.stream, row.serial, owned)
        self.received += 1
        self.physical_frames += row.physical_frames
        for ready in self.ingress.drain():
            if ready.key in self.rows:
                raise ValueError("Duplicate physical replay row")
            self.rows[ready.key] = ready
            self.accepted += 1
            while len(self.rows) > self.capacity:
                self.rows.popitem(last=False)

    def annotate(self, key, reward, events):
        """Keep fresh action-time reward/event labels separately from replay targets."""
        changes = dict(real_reward=float(reward), real_events=copy.deepcopy(events))
        if key in self.rows:
            self.rows[key] = replace(self.rows[key], **changes)
            return
        for serial, row in self.ingress.waiting.get(key[0], {}).items():
            if row.key == key:
                self.ingress.waiting[key[0]][serial] = replace(row, **changes)
                return

    def sequence(self, key, washout, future=1):
        row = self.rows[key]
        start = max(0, row.index - washout)
        prefix = []
        for index in range(start, row.index + 1):
            item = self.rows.get((row.stream, row.episode, index))
            if item is None:
                raise ValueError("missing_history")
            if index < row.index and (item.terminated or item.truncated):
                raise ValueError("cross_episode")
            prefix.append(item)
        suffix = []
        for index in range(row.index, row.index + future):
            item = self.rows.get((row.stream, row.episode, index))
            if item is None:
                break
            suffix.append(item)
            if item.terminated or item.truncated:
                break
        return prefix[:-1], suffix

    def reject(self, reason):
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def candidates(self, count):
        keys = list(self.rows)
        if not keys:
            return []
        indices = self.rng.integers(0, len(keys), size=count)
        return [keys[int(i)] for i in indices]

    def candidate_order(self, excluded=()):
        """One complete uniformly shuffled search of the current replay."""
        keys = [key for key in self.rows if key not in excluded]
        self.rng.shuffle(keys)
        return keys

    def state_dict(self):
        # Tensor/plain-container serialization keeps the established safe
        # weights-only place-field loader usable; no custom pickle classes.
        def encode(value):
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value)
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            return value

        rows = [{f.name: encode(getattr(row, f.name)) for f in fields(PhysicalDecision)} for row in self.rows.values()]
        return dict(
            schema="intrmotiv/physical-replay/v1",
            rows=rows,
            capacity=self.capacity,
            rng=copy.deepcopy(self.rng.bit_generator.state),
            received=self.received,
            accepted=self.accepted,
            physical_frames=self.physical_frames,
            session=self.session,
            rejected=dict(self.rejected),
            pending=self.ingress.pending,
        )

    def load_state_dict(self, state):
        if state.get("schema") != "intrmotiv/physical-replay/v1":
            raise ValueError("Unsupported physical replay checkpoint")

        def decode(value):
            if torch.is_tensor(value):
                return value.detach().cpu().numpy().copy()
            if isinstance(value, dict):
                return {k: decode(v) for k, v in value.items()}
            return value

        rows = [PhysicalDecision(**{k: decode(v) for k, v in item.items()}) for item in state["rows"]]
        self.rows = OrderedDict((row.key, row) for row in rows)
        self.capacity = state["capacity"]
        self.rng.bit_generator.state = state["rng"]
        for key in ("received", "accepted", "physical_frames", "rejected"):
            setattr(self, key, copy.deepcopy(state[key]))
        # DMLab itself is not restored. New actors start genuinely new episodes.
        self.session = int(state["session"]) + 1
        self.ingress = OrderedIngress(max_pending=self.capacity)
        if state["pending"]:
            self.rejected["restart_pending_tail"] = self.rejected.get("restart_pending_tail", 0) + state["pending"]
