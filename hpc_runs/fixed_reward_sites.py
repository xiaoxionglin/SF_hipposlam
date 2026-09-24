"""Build source-matched fixed-reward DMLab levels for the transfer study.

The source level already contains a transparent goal pickup with zero reward.
Moving that same entity and restoring its ten-point reward avoids introducing a
new visual object or a Python-side privileged-position reward path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


SITE_COLUMNS = {"dg50": 3, "dg51": 2}
SOURCE_LEVEL = "openfield_map2_fixed_loc3_noreward.lua"


def render_site(source: str, column: int) -> str:
    """Move the map-3 goal to top-row ``column`` without changing walls."""
    maps = list(re.finditer(r"\[\[\n([*PG A\n]+?)\n\]\]", source))
    if len(maps) < 3:
        raise ValueError("Expected the three fixed source maps")
    match = maps[2]
    rows = match.group(1).splitlines()
    if len(rows) != 21 or any(len(row) != 21 for row in rows):
        raise ValueError("Unexpected map-3 dimensions")
    goals = [(row, col) for row, text in enumerate(rows) for col, char in enumerate(text) if char == "G"]
    if len(goals) != 1 or rows[1][column] != "P":
        raise ValueError("Expected one old goal and a traversable reward-site spawn cell")
    old_row, old_col = goals[0]
    old = list(rows[old_row])
    old[old_col] = " "
    rows[old_row] = "".join(old)
    top = list(rows[1])
    top[column] = "G"  # also excludes this cell from the spawn distribution
    rows[1] = "".join(top)
    replacement = "\n".join(rows)
    result = source[: match.start(1)] + replacement + source[match.end(1) :]
    zero_reward = "  quantity = 0,"
    if result.count(zero_reward) != 1:
        raise ValueError("Expected the source's single zero-reward goal definition")
    return result.replace(zero_reward, "  quantity = 10,", 1)


def write_levels(source_path: Path, output_dir: Path) -> list[Path]:
    source = source_path.read_text()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for site, column in SITE_COLUMNS.items():
        path = output_dir / f"openfield_map2_fixed_reward_{site}.lua"
        path.write_text(render_site(source, column))
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Exact qualified source no-reward Lua level")
    parser.add_argument("output_dir", type=Path, help="DMLab levels directory in an isolated runtime")
    args = parser.parse_args()
    if args.source.name != SOURCE_LEVEL:
        parser.error(f"source must be {SOURCE_LEVEL}")
    for path in write_levels(args.source, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
