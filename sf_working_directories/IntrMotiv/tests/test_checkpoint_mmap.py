import numpy as np
import pytest
import torch

from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict


def test_mmap_checkpoint_values_aliases_and_private_storage(tmp_path):
    path = tmp_path / "checkpoint.pth"
    values = torch.arange(12, dtype=torch.float32)
    torch.save(
        {
            "model": {"weight": values.view(3, 4)},
            "view": values[3:6],
            "metadata": np.float64(1.25),
            "controller": {"replay": {"rows": [torch.arange(5)]}},
        },
        path,
    )
    state = load_checkpoint_dict(path, torch.device("cpu"))
    torch.testing.assert_close(state["model"]["weight"], values.view(3, 4))
    assert state["metadata"] == 1.25
    state["view"][0] = 99
    assert state["model"]["weight"][0, 3] == 99  # preserve original storage aliases
    fresh = load_checkpoint_dict(path, torch.device("cpu"))
    assert fresh["model"]["weight"][0, 3] == 3  # caller cannot alter checkpoint bytes
    torch.testing.assert_close(fresh["controller"]["replay"]["rows"][0], torch.arange(5))
