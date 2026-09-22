"""Export actual Lua-generated layouts selected by a validated StudySpec.

Run under Slurm on NEMO2, never on its login node. Native assets are selected
explicitly so this command cannot silently export an older installed level.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .geometry import entity_record
from .spec import load_study


def export(study_path: Path, runfiles: Path, output: Path, *, verify=False):
    import deepmind_lab

    study = load_study(study_path)
    deepmind_lab.set_runfiles_path(str(runfiles.resolve(strict=True)))
    settings = {}
    for run in study.expand_runs():
        args = dict(arg[2:].split("=", 1) for arg in run.args)
        seed = int(args["dmlab_map_seed"])
        opening = float(args["dmlab_wall_removal_probability"])
        settings[(seed, opening)] = args["env"]
    records = []
    for (seed, opening), level in sorted(settings.items()):
        lab = deepmind_lab.Lab(
            level,
            ["GEOMETRY.ENTITY_LAYER"],
            config={"geometrySeed": str(seed), "wallRemovalProbability": str(opening), "width": "96", "height": "72"},
            renderer="software",
        )
        try:
            lab.reset(seed=51000)
            entity = lab.observations()["GEOMETRY.ENTITY_LAYER"]
            if isinstance(entity, bytes):
                entity = entity.decode()
            records.append(entity_record(entity, map_seed=seed, wall_removal_probability=opening))
        finally:
            lab.close()
    generator = runfiles / "baselab/game_scripts/common/connected_maze.lua"
    archive = dict(
        schema="intrmotiv/map-archive/v1",
        generator="common.connected_maze/v1",
        generator_source_sha256=sha256(generator.read_bytes()).hexdigest(),
        maps=records,
    )
    if verify:
        if json.loads(output.read_text()) != archive:
            raise ValueError("Regenerated native geometry differs from the archived source/maps")
    else:
        if output.exists():
            raise FileExistsError("Archive exists; use --verify or a new output path")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(archive, indent=2) + "\n")
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    parser.add_argument("--runfiles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    archive = export(args.study, args.runfiles, args.output, verify=args.verify)
    print(f"Verified {len(archive['maps'])} native layouts")


if __name__ == "__main__":
    main()
