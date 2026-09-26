"""Version-consistent actor traces using original replay/core implementations."""

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from .controller_history import ControllerHistory, reconstruct_controller_history


@dataclass
class StreamMemory:
    episode: int
    version: int
    next_index: int = 0
    rows: deque = field(default_factory=deque)


class ActorMemory:
    def __init__(self):
        self.streams = {}
        self.rebuilds = self.failures = 0
        self.seconds = 0.0
        self.pending = None

    @torch.no_grad()
    def before_forward(self, model, raw_obs, state, version):
        ids = raw_obs["controller_identity"].detach().cpu().long().tolist()
        corrected = state.clone()
        self.pending = []
        try:
            published = getattr(getattr(model, "controller_q", None), "publication_version", None)
            if version < 0 or (published is not None and int(published.item()) != version):
                raise RuntimeError("Actor weights/publication version mismatch")
            for batch, (stream, episode, index, serial) in enumerate(ids):
                memory = self.streams.get(stream)
                if memory is None or memory.episode != episode:
                    if index != 0:
                        raise RuntimeError("Actor history missing at non-reset decision")
                    memory = StreamMemory(episode, version)
                    self.streams[stream] = memory
                if version < memory.version:
                    raise RuntimeError("Actor publication version regressed")
                if index != memory.next_index:
                    raise RuntimeError("Actor stream gap or duplicate decision")
                if memory.version != version and index:
                    started = time.perf_counter()
                    rows = list(memory.rows)
                    needed = min(index, model.core.expanded_length)
                    if len(rows) != needed:
                        raise RuntimeError("Incomplete actor reconstruction prefix")
                    obs = {key: torch.stack([row[1][key] for row in rows]) for key in rows[0][1]}
                    initial = state[batch : batch + 1]
                    h = ControllerHistory(
                        obs,
                        torch.tensor([row[0] for row in rows]),
                        initial,
                        torch.stack([row[2] for row in rows]).to(state.device),
                        rows[0][0] == 0,
                        0,
                    )
                    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
                    with torch.random.fork_rng(devices=devices):
                        rebuilt = reconstruct_controller_history(model, h, memory_only=True)["reconstructed_state"]
                    if hasattr(model.core, "split_worker_state"):
                        old_canonical, _ = model.core.split_worker_state(initial)
                        new_canonical, worker = model.core.split_worker_state(rebuilt)
                        merged = old_canonical.clone()
                        merged[:, : model.core.base_state_size] = new_canonical[:, : model.core.base_state_size]
                        corrected[batch : batch + 1] = model.core.join_worker_state(merged, worker)
                    else:
                        corrected[batch, : model.core.base_state_size] = rebuilt[0, : model.core.base_state_size]
                    self.rebuilds += 1
                    self.seconds += time.perf_counter() - started
                memory.version = version
                observation = {
                    k: v[batch].detach().cpu().clone()
                    for k, v in raw_obs.items()
                    if k not in getattr(model, "privileged_obs_keys", ()) and k != "controller_identity"
                }
                self.pending.append((memory, index, observation))
        except Exception:
            self.failures += 1
            raise
        return corrected

    def after_forward(self, model, conditions):
        if self.pending is None:
            raise RuntimeError("Missing actor pre-forward transaction")
        for batch, (memory, index, observation) in enumerate(self.pending):
            if getattr(model, "cfg", None) is not None and getattr(model.cfg, "controller_cache_visual", False):
                from .controller_transport import cached_observation

                observation = cached_observation(observation, model.encoder._controller_visual[batch].detach().cpu())
            memory.rows.append((index, observation, conditions[batch].detach().cpu().clone()))
            while len(memory.rows) > model.core.expanded_length:
                memory.rows.popleft()
            memory.next_index = index + 1
        self.pending = None
