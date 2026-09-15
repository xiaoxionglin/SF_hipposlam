from sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep import TARGET_FRAMES, select_checkpoints


def test_study_targets_use_existing_nearest_checkpoint_selection(tmp_path):
    directory = tmp_path / "checkpoint_p0" / "milestones"
    directory.mkdir(parents=True)
    files = []
    for i, frames in enumerate((327680, 150003712, 300007424)):
        path = directory / f"checkpoint_{i:09d}_{frames}.pth"
        path.touch()
        files.append(path)
    assert select_checkpoints(tmp_path, target_frames=[327680, 150000000, 300000000]) == list(
        zip([327680, 150000000, 300000000], files)
    )
    assert [t for t, p in select_checkpoints(tmp_path)] == list(TARGET_FRAMES)
