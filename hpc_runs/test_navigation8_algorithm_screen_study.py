import unittest
from pathlib import Path

from hpc_runs.intrmotiv_study import load_study
from hpc_runs.intrmotiv_study.sample_factory import build_run_description
from hpc_runs.intrmotiv_study.telemetry import (
    CheckpointRecord,
    build_intervention_manifest,
    build_place_field_manifests,
)

ROOT = Path(__file__).parent
STUDIES = ROOT / "studies"
PRODUCTION = STUDIES / "navigation8_algorithm_screen.study.json"
PREFLIGHT = STUDIES / "navigation8_algorithm_screen_preflight.study.json"


class Navigation8AlgorithmScreenStudyTests(unittest.TestCase):
    def setUp(self):
        self.study = load_study(PRODUCTION)
        self.runs = self.study.expand_runs()

    def test_production_matrix_and_action_interface(self):
        self.assertEqual(len(self.runs), 18)
        self.assertEqual({run.seed for run in self.runs}, {8, 99, 123})
        self.assertEqual(
            {run.base for run in self.runs},
            {
                "SCR_ARR_DIRS",
                "SAT_ARR_DIRO_FILM",
                "DGP_HIT_JOINT_LEG",
                "CPD_GATE_CA3_BPTT",
                "W_REF_STOP",
                "W_REF_JOINT",
            },
        )
        for run in self.runs:
            self.assertIn("--dmlab_navigation_action_set=True", run.args)
            self.assertIn("--dmlab_reduced_action_set=False", run.args)
            self.assertIn("--dmlab_extended_action_set=False", run.args)
            self.assertIn("--env_frameskip=4", run.args)
            self.assertIn("--train_for_env_steps=300000000", run.args)
            self.assertIn("--with_pos_obs=False", run.args)

    def test_selected_historical_mechanisms_are_preserved(self):
        by_base = {run.base: run for run in self.runs if run.seed == 99}
        self.assertIn("--dg_recruitment_endpoint_gate=silent", by_base["SCR_ARR_DIRS"].args)
        self.assertIn("--dg_recruitment_victim_rule=directional", by_base["SCR_ARR_DIRS"].args)
        self.assertIn("--dg_recruitment_endpoint_gate=open", by_base["SAT_ARR_DIRO_FILM"].args)
        self.assertIn("--hrl_goal_conditioning=target_id_film", by_base["SAT_ARR_DIRO_FILM"].args)
        self.assertIn("--hrl_control_outcome=target_hit", by_base["DGP_HIT_JOINT_LEG"].args)
        self.assertIn("--hrl_goal_conditioning=legacy", by_base["DGP_HIT_JOINT_LEG"].args)
        self.assertIn("--dg_context_feedback=gate", by_base["CPD_GATE_CA3_BPTT"].args)
        self.assertIn("--dg_context_history=ca3", by_base["CPD_GATE_CA3_BPTT"].args)
        self.assertIn("--dg_transition_prediction=none", by_base["CPD_GATE_CA3_BPTT"].args)
        for base, gradient in (("W_REF_STOP", "stop"), ("W_REF_JOINT", "joint")):
            run = by_base[base]
            self.assertIn(f"--ppo_dg_gradient={gradient}", run.args)
            self.assertIn("--intrinsic_goal_initialize_live_from_reference=True", run.args)
            reference = next(arg for arg in run.args if arg.startswith("--intrinsic_goal_reference_checkpoint="))
            self.assertIn("checkpoint_000004580_75038720.pth", reference)

    def test_historical_arguments_change_only_at_declared_batch_boundaries(self):
        sources = {
            "SCR_ARR_DIRS": ("source_credit_retirement.study.json", "SCR_C15_ARR_DIRS_S123"),
            "SAT_ARR_DIRO_FILM": ("saturday_batch.study.json", "SAT_C15_ARR_DIRO_FILM_S8"),
            "DGP_HIT_JOINT_LEG": (
                "dg_policy_gradient_first_outcome.study.json",
                "DGP_C15_HIT_JOINT_LEG_S123",
            ),
            "CPD_GATE_CA3_BPTT": (
                "ca3_feedback_predictive_dg.study.json",
                "CPD_C15_GATE_CA3_BPTT_S99",
            ),
            "W_REF_STOP": ("persistent_intrinsic_control.study.json", "PIC_W_REF_STOP_S8"),
            "W_REF_JOINT": ("persistent_intrinsic_control.study.json", "PIC_W_REF_JOINT_S8"),
        }
        allowed = {
            "--seed",
            "--train_for_env_steps",
            "--env_frameskip",
            "--dmlab_extended_action_set",
            "--dmlab_reduced_action_set",
            "--wandb_project",
            "--wandb_group",
            "--online_spatial_snapshot_targets",
            "--online_spatial_snapshot_max_frames",
        }

        def arg_map(args):
            return {arg.split("=", 1)[0]: arg for arg in args}

        current = {run.base: run for run in self.runs if run.seed == 99}
        for base, (filename, source_name) in sources.items():
            source_study = load_study(STUDIES / filename)
            source = next(run for run in source_study.expand_runs() if run.name == source_name)
            source_args = arg_map(source.args)
            current_args = arg_map(current[base].args)
            for flag in source_args.keys() - allowed:
                self.assertEqual(current_args.get(flag), source_args[flag], (base, flag))

    def test_telemetry_selects_42_field_runs_and_six_matched_goal_probes(self):
        inventory = [
            CheckpointRecord(
                run.name,
                target,
                target,
                Path(self.study.workspace_root) / run.name / f"checkpoint_{target}.pth",
                Path(self.study.workspace_root) / run.name,
            )
            for run in self.runs
            for target in self.study.telemetry["target_frames"]
        ]
        rows, trajectory = build_place_field_manifests(self.study, inventory, require_checkpoint_files=False)
        self.assertEqual(len(rows), 42)
        self.assertEqual(len(trajectory), 30)
        interventions = build_intervention_manifest(self.study, rows)
        self.assertEqual(len(interventions), 6)
        self.assertEqual(
            {row["condition"] for row in interventions},
            {"N8_W_REF_STOP", "N8_W_REF_JOINT"},
        )

    def test_production_and_preflight_are_submit_ready(self):
        production = build_run_description(self.study)
        self.assertEqual(len(production.experiments), 18)
        preflight = load_study(PREFLIGHT)
        preflight_runs = preflight.expand_runs()
        self.assertEqual(len(preflight_runs), 6)
        self.assertEqual({run.seed for run in preflight_runs}, {99})
        for run in preflight_runs:
            self.assertIn("--train_for_env_steps=2000000", run.args)
            self.assertIn("--online_spatial_snapshot_interval=1000000", run.args)
            self.assertIn("--online_spatial_snapshot_max_frames=2000000", run.args)
        self.assertEqual(len(build_run_description(preflight).experiments), 6)


if __name__ == "__main__":
    unittest.main()
