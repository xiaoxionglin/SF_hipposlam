"""Compatibility tests for the optional W&B integration."""

from types import SimpleNamespace

from sample_factory.utils import wandb_utils


class _Config(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def test_init_wandb_does_not_pass_removed_start_method(monkeypatch, tmp_path):
    settings_calls = []
    init_calls = []
    updates = []
    metrics = []
    fake_wandb = SimpleNamespace(
        Settings=lambda **kwargs: settings_calls.append(kwargs) or object(),
        init=lambda **kwargs: init_calls.append(kwargs),
        config=SimpleNamespace(update=lambda *args, **kwargs: updates.append((args, kwargs))),
        define_metric=lambda *args, **kwargs: metrics.append((args, kwargs)),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", fake_wandb)
    monkeypatch.setattr(wandb_utils, "wandb_dir", lambda cfg, mkdir: str(tmp_path))
    cfg = _Config(
        with_wandb=True,
        experiment="run",
        env="env",
        wandb_group="group",
        wandb_project="project",
        wandb_user="entity",
        wandb_job_type="train",
        wandb_tags=["condition"],
        wandb_step_metric_namespaces=(),
    )

    wandb_utils.init_wandb(cfg)

    assert settings_calls == [{}]
    assert init_calls[0]["settings"] is not None
    assert updates and metrics
