"""Rendering contracts: boundaries, visible silence, honest scales and exports."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from hpc_runs.intrmotiv_study import spatial


class AtlasFiguresTest(unittest.TestCase):
    def payload(self):
        return dict(
            pose=np.array([[1, 1, 0], [2, 2, 90], [8, 8, 180], [9, 9, 270]]),
            segment_id=np.array([0, 0, 1, 1]),
            dones=np.zeros(4, dtype=bool),
            dg_activity=np.array([[1, 0], [2, 0], [1, 0], [0, 0]]),
            bounds=np.array([0, 10, 0, 10]),
            grain=2,
            run_name="test",
            target_env_steps=5,
            actual_env_steps=6,
        )

    def test_segments_split_terminals_and_reused_ids(self):
        payload = self.payload()
        payload["segment_id"] = np.array([0, 0, 0, 0])
        payload["dones"][1] = True
        self.assertEqual(spatial.trajectory_segment_slices(payload), [slice(0, 2), slice(2, 4)])
        payload["segment_id"] = np.array([0, 1, 0, 0])
        self.assertEqual(spatial.trajectory_segment_slices(payload), [slice(0, 1), slice(1, 2), slice(2, 4)])

    def test_fields_keep_silent_units_and_common_scale(self):
        def inspect(fig, stem, plt):
            axes = [ax for ax in fig.axes if ax.images]
            self.assertEqual(len(axes), 2)
            self.assertIn("silent", axes[1].get_title())
            for ax in axes:
                self.assertEqual(ax.images[0].get_clim(), (0, 1))
            plt.close(fig)
            return []

        with patch.object(spatial, "_save_figure", side_effect=inspect):
            spatial.render_place_field_contact_sheets(self.payload(), Path("/tmp/test_fields"))

    def test_graph_masks_unattempted_without_masking_failure(self):
        def inspect(fig, stem, plt):
            image = fig.axes[0].images[0]
            self.assertEqual(image.get_clim(), (0, 1))
            self.assertTrue(image.get_array().mask[0, 0])
            self.assertFalse(image.get_array().mask[0, 1])
            self.assertEqual(image.get_array()[0, 1], 0)
            plt.close(fig)
            return []

        with patch.object(spatial, "_save_figure", side_effect=inspect):
            spatial.render_graph_outcomes(
                np.array([[0, 2], [2, 0]]), np.array([[0, 0], [1, 0]]), Path("/tmp/test_graph"), title="Test"
            )

    def test_svg_graph_preserves_explicit_stored_evidence_label(self):
        import xml.etree.ElementTree as ET

        import matplotlib.pyplot as plt

        with tempfile.TemporaryDirectory() as folder, plt.rc_context({"svg.fonttype": "none"}):
            outputs = spatial.render_graph_outcomes(
                np.array([[0, 2], [2, 0]]),
                np.array([[0, 1], [2, 0]]),
                Path(folder) / "stored",
                title="Stored graph",
                ratio_label="Stored confidence / attempts",
                figure_scale=0.58,
                formats=("svg",),
            )
            self.assertEqual([p.suffix for p in outputs], [".svg"])
            text = outputs[0].read_text()
            ET.fromstring(text)
            self.assertIn("Stored confidence / attempts", text)
            self.assertNotIn("Prospective hits / attempts", text)

    def test_trajectory_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            for renderer in (spatial.render_occupancy_trajectory, spatial.render_trajectory_segments):
                outputs = renderer(self.payload(), Path(folder) / renderer.__name__, title="Condition · seed 8")
                self.assertEqual({p.suffix for p in outputs}, {".png", ".pdf"})
                self.assertTrue(all(p.stat().st_size > 1000 for p in outputs))

    def test_overview_preserves_segments_and_distinct_colors(self):
        from matplotlib.collections import LineCollection

        def inspect(fig, stem, plt):
            ax = fig.axes[1]
            paths = next(item for item in ax.collections if isinstance(item, LineCollection))
            self.assertEqual(len(paths.get_segments()), 2)
            np.testing.assert_array_equal(paths.get_segments()[0], [[1, 1], [2, 2]])
            np.testing.assert_array_equal(paths.get_segments()[1], [[8, 8], [9, 9]])
            self.assertFalse(np.array_equal(paths.get_colors()[0], paths.get_colors()[1]))
            self.assertIn("DMLab", ax.get_xlabel())
            plt.close(fig)
            return []

        with patch.object(spatial, "_save_figure", side_effect=inspect):
            spatial.render_occupancy_trajectory(self.payload(), Path("/tmp/test_overview"))


if __name__ == "__main__":
    unittest.main()
