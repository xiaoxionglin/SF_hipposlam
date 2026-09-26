"""Tests for the G500 resource probe contract."""

from types import SimpleNamespace

from hpc_runs.hosts.g500 import profile_training


def test_resources_reports_host_gpu_and_pressure(monkeypatch):
    monkeypatch.setattr(profile_training.psutil, "cpu_percent", lambda: 12.5)
    monkeypatch.setattr(
        profile_training.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=96 * 2**30),
    )
    monkeypatch.setattr(profile_training.os, "getloadavg", lambda: (1.0, 2.0, 3.0))
    monkeypatch.setattr(
        profile_training.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="0, 80000, 10, 200\n1, 79000, 20, 210\n"),
    )
    monkeypatch.setattr(profile_training.Path, "read_text", lambda self: "some avg10=0.00")

    result = profile_training.resources()

    assert result["cpu_percent"] == 12.5
    assert result["available_ram_gib"] == 96
    assert result["gpus"][1] == {
        "index": 1.0,
        "free_mib": 79000.0,
        "utilization": 20.0,
        "power_w": 210.0,
    }
    assert set(result["pressure"]) == {"cpu", "memory", "io"}
