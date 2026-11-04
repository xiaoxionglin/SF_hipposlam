"""Opt-in workflow controls preserve legacy artifacts and discovery behavior."""

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from hpc_runs.intrmotiv_study.cli import build_parser
from hpc_runs.intrmotiv_study.discovery import discover_run_directories
from hpc_runs.intrmotiv_study.spatial import _save_figure, render_graph_outcomes
from hpc_runs.intrmotiv_study.spec import SpecError, load_study


class DiscoveryModeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        raw = json.loads((Path(__file__).parent / "studies/graph_stabilized_recruitment.study.json").read_text())
        raw["bases"] = raw["bases"][:1]
        raw["seeds"] = [99]
        raw["expected_runs"] = 1
        for factor in raw["factors"]:
            factor["levels"] = factor["levels"][:1]
        raw["training"]["workspace_root"] = str(self.root)
        raw["training"]["output_root"] = str(self.root / "batch")
        raw["analysis"] = {
            "step_tag": "steps",
            "terminal_width": 10,
            "window_metrics": {"score": "score"},
            "max_workers": 1,
        }
        raw["telemetry"] = {
            "protocol": "dg-place-fields-v1",
            "target_frames": [10],
            "trajectory_seed": 99,
            "terminal_seeds": [],
        }
        self.spec = self.root / "study.json"
        self.spec.write_text(json.dumps(raw))
        self.study = load_study(self.spec)
        self.run = self.study.expand_runs()[0]
        self.batch = self.root / "batch"
        self.actual = self.batch / (self.run.name + "_") / ("00_" + self.run.name)
        self.actual.mkdir(parents=True)
        (self.actual / "config.json").write_text("{}")
        checkpoints = self.actual / "checkpoint_p0"
        checkpoints.mkdir()
        (checkpoints / "checkpoint_000001_10.pth").write_bytes(b"selector fixture; never loaded")
        summary = self.actual / ".summary/0"
        summary.mkdir(parents=True)
        writer = EventFileWriter(str(summary))
        for step in (5, 10):
            writer.add_event(
                Event(
                    wall_time=float(step),
                    step=step,
                    summary=Summary(
                        value=[
                            Summary.Value(tag="steps", simple_value=step),
                            Summary.Value(tag="score", simple_value=step / 5),
                        ]
                    ),
                )
            )
        writer.close()

    def add_artifact(self):
        (self.actual.parent / "analysis/online_spatial/unbatched" / self.run.name).mkdir(parents=True)

    def cli(self, command, output, mode=None):
        tokens = [command, str(self.spec), str(self.batch), str(output)]
        if mode is not None:
            tokens.append("--run-discovery-mode=" + mode)
        args = build_parser().parse_args(tokens)
        args.func(args)

    def test_legacy_is_default_and_training_still_rejects_real_duplicates(self):
        expected = {self.run.name: self.actual}
        self.assertEqual(discover_run_directories(self.study, self.batch), expected)
        self.assertEqual(discover_run_directories(self.study, self.batch, run_discovery_mode="training"), expected)
        self.add_artifact()
        with self.assertRaises(SpecError):
            discover_run_directories(self.study, self.batch, run_discovery_mode="legacy")
        with self.assertRaises(SpecError):
            discover_run_directories(self.study, self.batch)
        self.assertEqual(discover_run_directories(self.study, self.batch, run_discovery_mode="training"), expected)
        duplicate = self.batch / "copy" / self.run.name
        duplicate.mkdir(parents=True)
        for mode in ("legacy", "training"):
            with self.assertRaises(SpecError):
                discover_run_directories(self.study, self.batch, run_discovery_mode=mode)
        with self.assertRaisesRegex(SpecError, "must be legacy or training"):
            discover_run_directories(self.study, self.batch, run_discovery_mode="typo")

    def test_cli_collectors_route_mode_and_keep_default_outputs(self):
        fingerprint = self.study.fingerprint
        commands = {
            "collect-online": ("per_run.csv", "condition_summary.csv", "analysis_manifest.json"),
            "render-telemetry": ("analysis_manifest.tsv", "trajectory_manifest.tsv", "study_manifest.json"),
        }
        for command, names in commands.items():
            default = self.root / (command + "-default")
            legacy = self.root / (command + "-legacy")
            self.cli(command, default)
            self.cli(command, legacy, "legacy")
            for name in names:
                self.assertEqual((default / name).read_bytes(), (legacy / name).read_bytes())
        self.add_artifact()
        for command, names in commands.items():
            with self.assertRaises(SpecError):
                self.cli(command, self.root / (command + "-blocked"))
            output = self.root / (command + "-training")
            self.cli(command, output, "training")
            baseline = self.root / (command + "-default")
            for name in names[:-1]:
                self.assertEqual((baseline / name).read_bytes(), (output / name).read_bytes())
            manifest = json.loads((output / names[-1]).read_text())
            metadata = manifest["collection"] if command == "collect-online" else manifest
            self.assertEqual(metadata.pop("run_discovery_mode"), "training")
            self.assertEqual(manifest, json.loads((baseline / names[-1]).read_text()))
        self.assertEqual(load_study(self.spec).fingerprint, fingerprint)


class GraphExportOptionTests(unittest.TestCase):
    def test_default_save_calls_match_legacy_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as folder:
            stem = Path(folder) / "figure"
            fig, plt = Mock(), Mock()
            self.assertEqual(_save_figure(fig, stem, plt), [stem.with_suffix(".png"), stem.with_suffix(".pdf")])
            self.assertEqual(
                fig.savefig.call_args_list,
                [
                    call(stem.with_suffix(".png"), dpi=100, bbox_inches="tight", facecolor="white"),
                    call(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white"),
                ],
            )
            plt.close.assert_called_once_with(fig)

    def test_cli_graph_exports_requested_svg_and_evidence_label(self):
        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as folder, plt.rc_context({"svg.fonttype": "none"}):
            root = Path(folder)
            path = root / "counts.npz"
            np.savez(path, attempts=[[0, 2], [2, 0]], successes=[[0, 1], [0, 0]])
            args = build_parser().parse_args(
                [
                    "render-graph",
                    str(path),
                    str(root / "graph"),
                    "--attempts-key=attempts",
                    "--successes-key=successes",
                    "--title=Stored graph",
                    "--ratio-label=Stored hits / attempts",
                    "--figure-scale=0.8",
                    "--format=svg",
                ]
            )
            args.func(args)
            svg = (root / "graph.svg").read_text()
            ET.fromstring(svg)
            self.assertIn("Stored hits / attempts", svg)
            self.assertNotIn("Prospective hits / attempts", svg)
            self.assertFalse((root / "graph.png").exists())
            self.assertFalse((root / "graph.pdf").exists())

    def test_invalid_export_options_fail_before_rendering(self):
        counts = np.zeros((2, 2))
        for options in (
            {"figure_scale": 0},
            {"figure_scale": float("nan")},
            {"formats": ()},
            {"formats": "svg"},
            {"formats": ("svg", "svg")},
            {"formats": ("../png",)},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                render_graph_outcomes(counts, counts, Path("unused"), title="Invalid", **options)

    def test_cli_requires_npz_and_named_matrices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.save(root / "array.npy", np.zeros((2, 2)))
            np.savez(root / "wrong_keys.npz", other=np.zeros((2, 2)))
            for filename in ("array.npy", "wrong_keys.npz"):
                args = build_parser().parse_args(
                    [
                        "render-graph",
                        str(root / filename),
                        str(root / "figure"),
                        "--attempts-key=attempts",
                        "--successes-key=successes",
                        "--title=Invalid input",
                    ]
                )
                with self.assertRaises(SpecError):
                    args.func(args)
            self.assertFalse(list(root.glob("figure.*")))


if __name__ == "__main__":
    unittest.main()
