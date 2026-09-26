import unittest
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study


class DGCapacityStudyTests(unittest.TestCase):
    def test_matrix_and_mechanisms(self):
        study = load_study(Path(__file__).with_name("studies") / "dg_capacity_goal_conditioning.study.json")
        runs = study.expand_runs()
        self.assertEqual(len(runs), 27)
        self.assertEqual({r.seed for r in runs}, {8, 99, 123})
        for r in runs:
            a = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in r.args}
            self.assertIn(a["--Hippo_n_feature"], ["16", "32", "64"])
            self.assertEqual(a["--env_frameskip"], "4")
            self.assertEqual(a["--dmlab_navigation_action_set"], "True")
            self.assertEqual(a["--train_for_env_steps"], "300000000")
            self.assertEqual(a["--dg_orthogonal_recruitment"], "False")
            self.assertEqual(a["--hrl_goal_conditioning"], "target_id_film")
            self.assertEqual(a["--ppo_dg_gradient"], "stop")
            self.assertEqual(a["--dg_goal_input"], "none" if r.base == "DIRECT_WORKER" else "write")
            self.assertEqual(a["--hrl_edge_exploration"], "True" if r.base == "WAYPOINT_DG" else "False")
            self.assertEqual(
                a["--hrl_manager_mode"], "frontier_waypoint" if r.base == "WAYPOINT_DG" else "frontier_direct"
            )

    def test_two_intervention_checkpoints_for_every_run(self):
        from hpc_runs.intrmotiv_study.spec import SpecError
        from hpc_runs.intrmotiv_study.telemetry import (
            CheckpointRecord,
            build_intervention_manifest,
            build_place_field_manifests,
        )

        study = load_study(Path(__file__).with_name("studies") / "dg_capacity_goal_conditioning.study.json")
        root = Path(study.output_root)
        inventory = [
            CheckpointRecord(r.name, t, t, root / r.name / f"checkpoint_1_{t}.pth", root / r.name)
            for r in study.expand_runs()
            for t in study.telemetry["target_frames"]
        ]
        rows, _ = build_place_field_manifests(study, inventory, require_checkpoint_files=False)
        selected = build_intervention_manifest(study, rows)
        self.assertEqual(len(selected), 54)
        self.assertEqual({int(r["target_frames"]) for r in selected}, {75000000, 300000000})
        with self.assertRaises(SpecError):
            build_intervention_manifest(study, selected[:-1])
        with self.assertRaises(SpecError):
            build_intervention_manifest(study, selected + [selected[0]])
        study.telemetry["intervention"]["target_frames"] = [300000000, 75000000]
        with self.assertRaises(SpecError):
            build_intervention_manifest(study, rows)

    def test_preflight_covers_every_cell(self):
        s = load_study(Path(__file__).with_name("studies") / "dg_capacity_goal_conditioning_preflight.study.json")
        runs = s.expand_runs()
        self.assertEqual(len(runs), 9)
        self.assertEqual({r.seed for r in runs}, {99})
        for r in runs:
            self.assertIn("--train_for_env_steps=2000000", r.args)
            self.assertIn("--keep_checkpoints=100", r.args)


if __name__ == "__main__":
    unittest.main()
