"""Prepare isolated DMLab assets without modifying the installed package.

Run in the same Python environment/binding as training. Links are immutable
release inputs; caches and compiled maps must live in the workspace separately.
"""

from pathlib import Path
import argparse
import json


def prepare_runfiles(source: Path, destination: Path) -> Path:
    import deepmind_lab

    source = source.resolve(strict=True)
    base = Path(deepmind_lab.runfiles_path()).resolve(strict=True)
    destination = destination.resolve()
    scripts = source / "deepmindlab_patch/game_scripts"
    if not scripts.is_dir():
        raise ValueError(f"Missing game scripts: {scripts}")
    if destination == base:
        raise ValueError("Cannot replace installed DMLab assets")
    destination.mkdir(parents=True, exist_ok=True)

    def link(target, path):
        if path.is_symlink():
            if path.resolve() != target.resolve():
                raise ValueError(f"Existing runfiles link differs: {path}")
        elif path.exists():
            raise ValueError(f"Runfiles destination already exists: {path}")
        else:
            path.symlink_to(target)

    for entry in base.iterdir():
        if entry.name != "baselab":
            link(entry, destination / entry.name)
    (destination / "baselab").mkdir(exist_ok=True)
    for entry in (base / "baselab").iterdir():
        if entry.name != "game_scripts":
            link(entry, destination / "baselab" / entry.name)
    link(scripts, destination / "baselab/game_scripts")
    (destination / "provenance.json").write_text(
        json.dumps(dict(source=str(source), native_assets=str(base)), indent=2) + "\n"
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_runfiles(args.source, args.output))


if __name__ == "__main__":
    main()
