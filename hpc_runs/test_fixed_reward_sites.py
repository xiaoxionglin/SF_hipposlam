"""Source-geometry checks for the two fixed-reward transfer levels."""

from pathlib import Path
import re
import unittest

from hpc_runs.fixed_reward_sites import SITE_COLUMNS, render_site


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "deepmindlab_patch/game_scripts/levels/openfield_map2_fixed_loc3_noreward.lua"
if not SOURCE.exists():
    SOURCE = Path(
        "/home/xiaoxiong/SFgit/SF_hipposlam/deepmindlab_patch/game_scripts/levels/"
        "openfield_map2_fixed_loc3_noreward.lua"
    )


def _maps(text: str) -> list[list[str]]:
    return [match.splitlines() for match in re.findall(r"\[\[\n([*PG A\n]+?)\n\]\]", text)]


class FixedRewardSiteTests(unittest.TestCase):
    def test_goal_moves_without_changing_floor_or_walls(self):
        if not SOURCE.exists():
            self.skipTest("Desktop runtime source is unavailable")
        source = SOURCE.read_text()
        source_maps = _maps(source)
        for site, column in SITE_COLUMNS.items():
            with self.subTest(site=site):
                result = render_site(source, column)
                maps = _maps(result)
                self.assertEqual(maps[:2], source_maps[:2])
                self.assertEqual(maps[2][1][column], "G")
                self.assertEqual(sum(row.count("G") for row in maps[2]), 1)
                self.assertEqual(
                    [[char == "*" for char in row] for row in maps[2]],
                    [[char == "*" for char in row] for row in source_maps[2]],
                )
                self.assertEqual(result.count("  quantity = 10,"), 1)
                self.assertEqual(result.count("models/goal_transparent.md3"), 1)


if __name__ == "__main__":
    unittest.main()
