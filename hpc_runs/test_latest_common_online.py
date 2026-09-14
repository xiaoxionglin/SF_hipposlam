"""Real TensorBoard fixtures for shared-step collection."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from hpc_runs.intrmotiv_study import SpecError
from hpc_runs.intrmotiv_study.tensorboard import collect_online_records


class LatestCommonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        runs = [
            SimpleNamespace(name=n, condition=n, base=n, seed=1, factors={}, metadata={})
            for n in ("control", "transfer")
        ]
        self.study = SimpleNamespace(
            analysis={
                "step_tag": "steps",
                "terminal_width": 10,
                "window_metrics": {"score": "score"},
                "max_workers": 2,
                "scalar_size_guidance": 1,
            },
            expand_runs=lambda: runs,
        )

    def write(self, name, steps, scores):
        folder = self.root / name / ".summary" / "0"
        folder.mkdir(parents=True)
        writer = EventFileWriter(str(folder))
        for tag, entries in [("steps", [(s, s) for s in steps]), ("score", scores)]:
            for step, value in entries:
                writer.add_event(
                    Event(
                        wall_time=float(step),
                        step=step,
                        summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)]),
                    )
                )
        writer.close()

    def test_aligns_to_metric_coverage_reads_once_and_keeps_all_scalars(self):
        self.write("control", [10, 20, 30], [(10, 1), (15, 3), (20, 5), (30, 100)])
        self.write("transfer", [10, 20, 40], [(10, 2), (15, 4), (20, 6)])
        calls = []
        reload = EventAccumulator.Reload

        def record(acc):
            calls.append(acc)
            return reload(acc)

        with patch.object(EventAccumulator, "Reload", record):
            rows = collect_online_records(self.study, self.root, latest_common=True)
        self.assertEqual(len(calls), 2)
        self.assertEqual([(r["window_low"], r["window_high"]) for r in rows], [(10, 20)] * 2)
        self.assertEqual([r["max_step"] for r in rows], [30, 40])
        self.assertEqual([r["score"] for r in rows], [3, 4])
        self.assertEqual([r["score__n"] for r in rows], [3, 3])

    def test_missing_metric_fails(self):
        self.write("control", [10, 20], [(10, 1), (20, 2)])
        self.write("transfer", [10, 20], [])
        with self.assertRaisesRegex(SpecError, "required histories"):
            collect_online_records(self.study, self.root, latest_common=True)

    def test_empty_common_metric_window_fails(self):
        self.write("control", [10, 20], [(20, 1)])
        self.write("transfer", [10, 40], [(30, 2), (40, 4)])
        with self.assertRaisesRegex(SpecError, "no finite"):
            collect_online_records(self.study, self.root, latest_common=True)

    def test_fixed_and_terminal_modes_preserved(self):
        self.study.analysis["scalar_size_guidance"] = 30000
        self.write("control", [10, 20, 30], [(10, 1), (20, 2), (30, 3)])
        self.write("transfer", [10, 20, 40], [(10, 4), (20, 5), (40, 6)])
        rows = collect_online_records(self.study, self.root)
        self.assertEqual([r["window_high"] for r in rows], [30, 40])
        rows = collect_online_records(self.study, self.root, fixed_window=(10, 20))
        self.assertEqual([r["score"] for r in rows], [1.5, 4.5])

    def test_conflicting_window_rejected(self):
        with self.assertRaisesRegex(SpecError, "mutually exclusive"):
            collect_online_records(self.study, self.root, (1, 2), latest_common=True)


if __name__ == "__main__":
    unittest.main()
