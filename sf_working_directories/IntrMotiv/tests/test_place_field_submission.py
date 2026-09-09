from pathlib import Path

import pytest

from sf_working_directories.IntrMotiv.evaluation.submit_place_field_sweep import (
    ManifestRow,
    build_sbatch_command,
    parse_job_id,
    selected_indices,
)


def test_single_submission_command_has_no_array_or_dependency():
    row = ManifestRow(33, {"label_suffix": "c05_s99_100m"})
    command = build_sbatch_command(
        row=row,
        manifest=Path("/work/classic/fr_xl1014-train/manifest.tsv"),
        output_dir=Path("/work/classic/fr_xl1014-train/analysis/c05"),
        runner=Path("/source/run_place_field_sweep_single.sh"),
        partition="cpu",
        cpus=4,
        memory="16G",
        time_limit="00:40:00",
        max_num_frames=10000,
        job_name_prefix="pf",
        replay_observation_panel=Path("/work/classic/fr_xl1014-train/panel.npz"),
    )

    assert command[0] == "sbatch"
    assert not any(argument.startswith("--array") for argument in command)
    assert not any(argument.startswith("--dependency") for argument in command)
    assert command[-3:] == [
        "/work/classic/fr_xl1014-train/manifest.tsv",
        "33",
        "/work/classic/fr_xl1014-train/analysis/c05",
    ]
    assert "PLACE_FIELD_MAX_FRAMES=10000" in command[7]
    assert "PLACE_FIELD_REPLAY_PANEL=/work/classic/fr_xl1014-train/panel.npz" in command[7]


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
