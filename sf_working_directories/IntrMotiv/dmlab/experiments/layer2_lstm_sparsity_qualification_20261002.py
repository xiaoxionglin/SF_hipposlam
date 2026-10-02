"""Two-arm layer-2 ResNet/LSTM sparsification qualification."""

from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description


SPEC = (
    Path(__file__).resolve().parents[4]
    / "hpc_runs/studies/layer2_lstm_sparsity_qualification_20261002.study.json"
)
STUDY = load_study(SPEC)
RUN_DESCRIPTION = build_run_description(STUDY)
