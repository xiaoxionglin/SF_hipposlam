"""Check the built wheel independently of source-checkout/editable imports.

Usage: python scripts/check_wheel_install.py dist/sample_factory-*.whl
The wheel is installed without dependencies/network into a temporary directory.
Existing interpreter dependencies are reused; no training or host probes run.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.machinery
import json
import runpy
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def probe(target: Path) -> None:
    target = target.resolve()
    paths = sysconfig.get_paths()
    dependency_roots = {Path(paths[key]).resolve() for key in ("stdlib", "platstdlib", "purelib", "platlib")}
    # -I ignores PYTHONPATH, but editable installs can still inject .pth paths
    # and meta-path finders. Keep only the interpreter/dependency paths and
    # ordinary import machinery so missing wheel files cannot be hidden.
    sys.path = [str(target)] + [
        entry
        for entry in sys.path
        if entry and any(_is_within(Path(entry).resolve(), root) for root in dependency_roots)
    ]
    sys.meta_path = [
        importlib.machinery.BuiltinImporter,
        importlib.machinery.FrozenImporter,
        importlib.machinery.PathFinder,
    ]
    modules = [
        "sf_shared.checkpoints",
        "sf_shared.depth",
        "sf_shared.dg",
        "sf_examples.simgay.train",
        "sf_working_directories.IntrMotiv.dmlab.controller_schedule",
        "sf_working_directories.IntrMotiv.dmlab.controller_replay",
        "sf_working_directories.IntrMotiv.dmlab.online_spatial_telemetry",
        "hpc_runs.intrmotiv_study.cli",
        "hpc_runs.intrmotiv_study.direct",
        "hpc_runs.intrmotiv_offpolicy.sf_transport",
        "hpc_runs.intrmotiv_offpolicy.train",
        "hpc_runs.offpolicy_goal_baselines.train_parallel",
        "hpc_runs.hosts.g500.profile_training",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert _is_within(Path(module.__file__).resolve(), target), (name, module.__file__)
    for name, module in list(sys.modules.items()):
        if name.startswith(("sample_factory.", "sf_shared.", "sf_examples.", "sf_working_directories.", "hpc_runs.")):
            location = getattr(module, "__file__", None)
            if location:
                assert _is_within(Path(location).resolve(), target), (name, location)
    assert "deepmind_lab" not in sys.modules
    assert "torchvision" not in sys.modules

    from hpc_runs.intrmotiv_study.geometry import geometry_from_config

    schema = json.loads((target / "hpc_runs/intrmotiv_study/study.schema.json").read_text())
    assert schema["properties"]["schema"]["const"] == "intrmotiv/study/v1"
    for asset, environment in (
        ("corridor_geometry", "corridor_geometry_noreward"),
        ("easy_landmark_maze", "easy_landmark_maze_noreward"),
    ):
        manifest = target / "hpc_runs/studies/assets" / asset / "maps.json"
        record = json.loads(manifest.read_text())["maps"][0]
        entity = record["entity_layer"].splitlines()
        config = SimpleNamespace(
            env=environment,
            dmlab_geometry_manifest="",
            dmlab_map_seed=record["map_seed"],
            dmlab_wall_removal_probability=record["wall_removal_probability"],
            dmlab_map_rows=len(entity),
            dmlab_map_cols=len(entity[0]),
            dmlab_cue_layout_seed=record.get("cue_layout_seed", 0),
            dmlab_landmark_cues=record.get("cue_mode", "none"),
        )
        assert geometry_from_config(config) == record
    try:
        sys.argv = ["hpc_runs.intrmotiv_study", "--help"]
        runpy.run_module("hpc_runs.intrmotiv_study", run_name="__main__")
    except SystemExit as error:
        assert error.code == 0
    print(f"Installed-wheel checks passed: {len(modules)} imports, study CLI, schema and both geometry defaults")


def check_wheel(wheel: Path) -> None:
    wheel = wheel.resolve()
    with zipfile.ZipFile(wheel) as archive:
        paths = archive.namelist()
        assert not any(name.startswith("hpc_runs/source_snapshots/") for name in paths)
        assert not any(name.startswith("hpc_runs/") and name.count("/") == 1 for name in paths)
    with tempfile.TemporaryDirectory(prefix="sf-wheel-check-") as temporary:
        root = Path(temporary)
        target = root / "installed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--no-cache-dir",
                "--disable-pip-version-check",
                "--target",
                str(target),
                str(wheel),
            ],
            cwd=root,
            check=True,
        )
        subprocess.run(
            [sys.executable, "-I", "-B", str(Path(__file__).resolve()), "--installed-target", str(target)],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("--installed-target", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.installed_target is not None:
        probe(args.installed_target)
    elif args.wheel is not None:
        check_wheel(args.wheel)
    else:
        parser.error("supply the built wheel path")
