from contextlib import nullcontext
from threading import Lock
from types import SimpleNamespace

import pytest
import torch

from sample_factory.algo.utils.model_sharing import ParameterClientAsync


@pytest.mark.parametrize(
    "device,mode,expected", [("cpu", "ddqn", 0), ("cuda", "ppo", 0), ("cuda", "ddqn", 1), ("cuda", "shadow", 1)]
)
def test_controller_gpu_copy_finishes_inside_weight_lock(monkeypatch, device, mode, expected):
    client = object.__new__(ParameterClientAsync)
    calls = []
    client.cfg = SimpleNamespace(controller_learning=mode, serial_mode=False)
    client.policy_id = 0
    client.policy_versions = torch.tensor([1])
    client.latest_policy_version = 0
    client.num_policy_updates = 0
    client._policy_lock = Lock()
    client.timing = SimpleNamespace(time_avg=lambda name: nullcontext())
    client._shared_model_weights = {}
    client._actor_critic = SimpleNamespace(
        load_state_dict=lambda weights: calls.append("copy"),
        parameters=lambda: iter([SimpleNamespace(device=torch.device(device))]),
    )

    def synchronize(device):
        assert client._policy_lock.locked()
        assert client.latest_policy_version == 0 and calls == ["copy"]
        calls.append("synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    client.ensure_weights_updated()
    assert client.latest_policy_version == 1 and not client._policy_lock.locked()
    assert calls.count("synchronize") == expected
