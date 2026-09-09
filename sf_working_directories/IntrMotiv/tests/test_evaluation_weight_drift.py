from pathlib import Path

import torch

from sf_working_directories.IntrMotiv.evaluation.weight_drift import Checkpoint, row_cosine, select_checkpoints


def test_row_cosine_identical_and_orthogonal_rows():
    reference = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    current = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
    result = row_cosine(current, reference)
    assert torch.allclose(result, torch.tensor([1.0, 0.0]))


def test_final_checkpoint_replaces_near_final_milestone_for_late_drift():
    checkpoints = [
        Checkpoint(Path("checkpoint_0.pth"), 0),
        Checkpoint(Path("checkpoint_75m.pth"), 75_000_000),
        Checkpoint(Path("checkpoint_100m_milestone.pth"), 100_000_000),
        Checkpoint(Path("checkpoint_final.pth"), 100_040_704),
    ]
    selected = select_checkpoints(checkpoints, [0, 75_000_000, 100_000_000])
    assert [checkpoint.env_steps for checkpoint in selected] == [0, 75_000_000, 100_040_704]
