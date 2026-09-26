"""Slurm-only native check of the five number-cued invisible reward sites."""

import argparse
from pathlib import Path

import deepmind_lab
import numpy as np

SITES = [(350, 1950), (250, 1950), (1550, 1250), (650, 1450), (850, 250)]
ACTIONS = [
    np.asarray(a, dtype=np.intc)
    for a in (
        (0, 0, 0, 1, 0, 0, 0),
        (0, 0, 0, -1, 0, 0, 0),
        (0, 0, -1, 0, 0, 0, 0),
        (0, 0, 1, 0, 0, 0, 0),
    )
]


def inside_cell(position, center):
    return all(center[i] - 50 <= position[i] < center[i] + 50 for i in (0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runfiles", required=True, type=Path)
    args = parser.parse_args()
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    lab = deepmind_lab.Lab(
        "openfield_map2_cued_reward5",
        ["INSTR", "DEBUG.POS.TRANS", "RGB_INTERLEAVED"],
        config={"width": "96", "height": "72"},
        renderer="software",
    )
    contacted = set()
    seen = set()
    try:
        for seed in range(1000, 5000):
            lab.reset(seed=seed)
            observation = lab.observations()
            instruction = int(observation["INSTR"])
            assert 1 <= instruction <= 5, (seed, instruction)
            seen.add(instruction)
            start = np.asarray(observation["DEBUG.POS.TRANS"][:2], dtype=float)
            assert all(not inside_cell(start, site) for site in SITES), (seed, instruction, start)
            if instruction in contacted or np.linalg.norm(start - SITES[instruction - 1]) >= 180:
                continue
            for _ in range(24):
                previous_distance = np.linalg.norm(
                    np.asarray(lab.observations()["DEBUG.POS.TRANS"][:2]) - SITES[instruction - 1]
                )
                for action in ACTIONS:
                    reward = lab.step(action, num_steps=4)
                    if reward > 0:
                        assert reward == 10 and not lab.is_running(), (instruction, seed, reward)
                        contacted.add(instruction)
                        print("contacted", instruction, "seed", seed, flush=True)
                        break
                    if not lab.is_running():
                        break
                    if (
                        np.linalg.norm(np.asarray(lab.observations()["DEBUG.POS.TRANS"][:2]) - SITES[instruction - 1])
                        < previous_distance
                    ):
                        break
                if not lab.is_running():
                    break
            if len(contacted) == 5:
                break
        assert seen == {1, 2, 3, 4, 5}, seen
        assert contacted == seen, ("Missing legal reward entries", contacted, seen)
        print("PASS: all five cues, spawn exclusions, +10 terminal contacts", flush=True)
    finally:
        lab.close()


if __name__ == "__main__":
    main()
