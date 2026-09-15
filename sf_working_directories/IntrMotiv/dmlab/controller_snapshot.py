"""Snapshot ownership for controller replay through the actual IntrMotiv model.

A snapshot owns every parameter and buffer influencing its value input, including
DG, worker-write modulation, normalization, and manager graph. It is constructed
through the original factory, not by copying a model with live autograd caches.
"""

from contextlib import contextmanager

import torch
from torch import nn


class ControllerSnapshot:
    def __init__(self, model_factory, source, version: int):
        if version < 0:
            raise ValueError("Snapshot version must be nonnegative")
        # Auxiliary/target construction must not perturb behavior RNG streams.
        devices = sorted({p.device.index for p in source.parameters() if p.is_cuda})
        with torch.random.fork_rng(devices=devices):
            self.model = model_factory()
        self.model.load_state_dict(
            {k: v.detach().clone() for k, v in source.state_dict().items()}, strict=True, assign=True
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self.version = int(version)

    @torch.no_grad()
    def refresh(self, source, version: int):
        if version <= self.version:
            raise ValueError("Snapshot refresh must advance its version")
        self.model.load_state_dict(source.state_dict(), strict=True)
        self.model.eval()
        self.version = int(version)


class _ReplayCall(nn.Module):
    def __init__(self, model, operation):
        super().__init__()
        self.model = model
        self.operation = operation

    def forward(self, *args):
        return self.operation(self.model, *args)


def differentiable_replay(snapshot, source, operation, *args, source_version: int):
    """Use live online parameters with immutable snapshot buffers and private caches.

    `operation` must use the supplied actual model. Gradients reach `source`
    parameters; replay forwards cannot mutate its graph, running statistics, or
    forward caches. Each call clones snapshot buffers so even an accidental
    in-place buffer update cannot become additional real-data graph evidence.
    Only use an online snapshot from the same parameter version as `source`.
    Target evaluation uses its own snapshot directly under no_grad instead.
    """
    if source_version != snapshot.version:
        raise ValueError("Online parameters and replay buffers must have the same version")
    if snapshot.model.training:
        raise RuntimeError("Replay snapshot must be in evaluation mode")
    wrapper = _ReplayCall(snapshot.model, operation)
    # STOP owns DG parameters only in the fresh-data optimizer transaction.
    # Detaching a slice of a concatenated head still creates zero DG gradients;
    # Adam would then apply its existing momentum during every replay update.
    projection = getattr(getattr(source, "encoder", None), "DG_projection", None)
    stop = getattr(getattr(source, "cfg", None), "ppo_dg_gradient", "joint") == "stop"
    stopped = {id(p) for p in projection.parameters()} if stop and projection is not None else set()
    parameters = {f"model.{k}": v.detach() if id(v) in stopped else v for k, v in source.named_parameters()}
    buffers = {f"model.{k}": v.detach().clone() for k, v in snapshot.model.named_buffers()}
    # strict matching catches target/online architecture drift, including HER.
    devices = sorted({p.device.index for p in source.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=devices):
        return torch.func.functional_call(wrapper, (parameters, buffers), args, strict=True)


@torch.no_grad()
def evaluate_replay(snapshot, operation, *args):
    """Evaluate a target with its own parameters and immutable cloned buffers."""
    return differentiable_replay(snapshot, snapshot.model, operation, *args, source_version=snapshot.version)


@contextmanager
def fresh_dg_parameter_owner(model):
    """Keep controller Adam state out of the fresh DG-only transaction."""
    owned = {id(p) for p in model.encoder.DG_projection.parameters()}
    suspended = [p for p in model.parameters() if p.requires_grad and id(p) not in owned]
    for parameter in suspended:
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter in suspended:
            parameter.requires_grad_(True)
