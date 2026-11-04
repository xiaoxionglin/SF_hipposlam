"""Offline checkpoint helpers retain their old API without evaluator imports."""

import subprocess
import sys
from pathlib import Path

from sf_shared.checkpoints import load_checkpoint_dict


def test_legacy_evaluator_reexports_same_loader():
    from sf_working_directories.IntrMotiv.evaluation.place_fields import load_checkpoint_dict as legacy_loader

    assert legacy_loader is load_checkpoint_dict


def test_checkpoint_helper_import_does_not_require_evaluation_stack():
    script = """
import importlib.abc
import sys

class BlockEvaluationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'deepmind_lab', 'torchvision', 'pandas', 'sample_factory', 'sf_working_directories'
        }:
            raise ImportError('Checkpoint helper imported unrelated dependency: ' + fullname)
        return None

sys.meta_path.insert(0, BlockEvaluationImports())
from sf_shared.checkpoints import load_checkpoint_dict
assert callable(load_checkpoint_dict)
assert 'sf_working_directories.IntrMotiv.evaluation.place_fields' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[2], check=True)
