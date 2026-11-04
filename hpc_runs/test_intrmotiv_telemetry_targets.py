import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.telemetry import discover_nemo_checkpoints


class TelemetryTargets(unittest.TestCase):
    def test_discovery_uses_study_and_intervention_targets_not_legacy_constants(self):
        study=load_study(Path(__file__).with_name('studies')/'full_system_controller_telemetry_probe.study.json')
        study.telemetry['intervention']={'target_frames':[150000000,300000000]}
        run=study.expand_runs()[0];directory=Path(study.workspace_root)/'fixture'
        requested=[]
        def select(run_dir,*,target_frames):
            requested.append((run_dir,list(target_frames)))
            return [(t,directory/f'checkpoint_000001_{t}.pth') for t in target_frames]
        module='sf_working_directories.IntrMotiv.evaluation.build_place_field_sweep'
        fake=SimpleNamespace(select_checkpoints=select,checkpoint_frames=lambda p:int(p.stem.rsplit('_',1)[1]))
        with patch.dict(sys.modules,{module:fake}),patch('hpc_runs.intrmotiv_study.telemetry.discover_run_directories',return_value={run.name:directory}):
            inventory=discover_nemo_checkpoints(study,directory)
        self.assertEqual(requested,[(directory,[327680,150000000,300000000])])
        self.assertEqual([r.target_frames for r in inventory],[327680,150000000,300000000])


if __name__=='__main__':unittest.main()
