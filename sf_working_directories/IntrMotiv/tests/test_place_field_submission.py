from pathlib import Path

import pytest

from sf_working_directories.IntrMotiv.evaluation.submit_place_field_sweep import (
    SOURCE_ROOT,
    ManifestRow,
    build_sbatch_command,
    parse_job_id,
    selected_indices,
)


def test_single_submission_command_has_no_array_or_dependency():
    row = ManifestRow(33, {"label_suffix": "c05_s99_100m"})
    command = build_sbatch_command(
        row=row,
        manifest=Path("/work/classic/fr_xl1014-corridor-geometry/manifest.tsv"),
        output_dir=Path("/work/classic/fr_xl1014-corridor-geometry/analysis/c05"),
        runner=Path("/source/run_place_field_sweep_single.sh"),
        partition="cpu",
        cpus=4,
        memory="16G",
        time_limit="00:40:00",
        max_num_frames=10000,
        job_name_prefix="pf",
        replay_observation_panel=Path("/work/classic/fr_xl1014-corridor-geometry/panel.npz"),
    )

    assert command[0] == "sbatch"
    assert not any(argument.startswith("--array") for argument in command)
    assert not any(argument.startswith("--dependency") for argument in command)
    assert command[-3:] == [
        "/work/classic/fr_xl1014-corridor-geometry/manifest.tsv",
        "33",
        "/work/classic/fr_xl1014-corridor-geometry/analysis/c05",
    ]
    assert f"INTRMOTIV_RUNTIME_SOURCE={SOURCE_ROOT}" in command[7]
    assert "INTRMOTIV_WORKSPACE_ROOT=/work/classic/fr_xl1014-corridor-geometry" in command[7]
    assert "PLACE_FIELD_MAX_FRAMES=10000" in command[7]
    assert "PLACE_FIELD_REPLAY_PANEL=/work/classic/fr_xl1014-corridor-geometry/panel.npz" in command[7]


def test_submission_command_propagates_workspace_override():
    workspace = Path("/work/classic/fr_xl1014-corridor-geometry")
    row = ManifestRow(0, {"label_suffix": "ca3_context"})
    command = build_sbatch_command(
        row=row,
        manifest=workspace / "manifest.tsv",
        output_dir=workspace / "analysis/ca3",
        runner=Path("/source/run_place_field_sweep_single.sh"),
        partition="cpu",
        cpus=4,
        memory="16G",
        time_limit="00:20:00",
        max_num_frames=500,
        job_name_prefix="pf",
        workspace_root=workspace,
    )

    assert f"INTRMOTIV_WORKSPACE_ROOT={workspace}" in command[7]


def test_row_selectors_are_sorted_unique_and_bounded():
    assert selected_indices(["3,1", "2-4", "3"], 6) == [1, 2, 3, 4]
    with pytest.raises(IndexError):
        selected_indices(["6"], 6)
    with pytest.raises(ValueError):
        selected_indices(["4-2"], 6)


def test_parse_job_id():
    assert parse_job_id("Submitted batch job 7975111\n") == "7975111"
    with pytest.raises(RuntimeError):
        parse_job_id("submission failed")


def test_episode_coverage_flags_exported_to_canonical_worker(tmp_path):
    from sf_working_directories.IntrMotiv.evaluation.submit_place_field_sweep import ManifestRow, build_sbatch_command

    row = ManifestRow(0, {"label_suffix": "test"})
    command = build_sbatch_command(
        row=row,
        manifest=tmp_path / "manifest.tsv",
        output_dir=tmp_path,
        runner=tmp_path / "worker.sh",
        partition="cpu",
        cpus=4,
        memory="16G",
        time_limit="01:00:00",
        max_num_frames=10000,
        job_name_prefix="test",
        coverage_episodes=100,
        random_coverage=True,
    )
    export = next(arg for arg in command if arg.startswith("--export="))
    assert "PLACE_FIELD_COVERAGE_EPISODES=100" in export
    assert "PLACE_FIELD_RANDOM_COVERAGE=1" in export


def test_alternate_workspace_is_enforced_and_exported(tmp_path, monkeypatch):
    from sf_working_directories.IntrMotiv.evaluation import submit_place_field_sweep as module

    monkeypatch.setattr(module, "WORKSPACE_ROOT", tmp_path)
    assert module.workspace_path(tmp_path / "output", "Output") == tmp_path / "output"
    with pytest.raises(ValueError, match="must be under"):
        module.workspace_path(tmp_path.parent / "outside", "Output")
    command = module.build_sbatch_command(
        row=ManifestRow(0, {"label_suffix": "workspace"}),
        manifest=tmp_path / "manifest.tsv",
        output_dir=tmp_path / "out",
        runner=tmp_path / "runner.sh",
        partition="cpu",
        cpus=4,
        memory="16G",
        time_limit="01:00:00",
        max_num_frames=10,
        job_name_prefix="test",
    )
    assert f"INTRMOTIV_WORKSPACE_ROOT={tmp_path}" in next(x for x in command if x.startswith("--export="))
