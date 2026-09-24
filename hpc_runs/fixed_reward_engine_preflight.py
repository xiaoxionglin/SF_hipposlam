"""Native DMLab smoke check for the two invisible fixed-reward pickups.

This is a qualification script, not a policy evaluation. DEBUG pose is read
only here to verify engine geometry and pickup behavior; training never requests
it as a policy observation.
"""

import argparse
from pathlib import Path

import deepmind_lab
import numpy as np


ACTIONS = [
    np.array(action, dtype=np.intc)
    for action in ((0, 0, 0, 1, 0, 0, 0), (0, 0, 0, -1, 0, 0, 0),
                   (0, 0, -1, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0, 0))
]
SITES = {"dg50": (350.0, 1950.0), "dg51": (250.0, 1950.0)}


def pose(lab):
    return np.asarray(lab.observations()["DEBUG.POS.TRANS"][:2], dtype=float)


def run_site(site):
    target = np.asarray(SITES[site])
    lab = deepmind_lab.Lab(
        "openfield_map2_fixed_reward_" + site,
        ["DEBUG.POS.TRANS"],
        config={"width": "96", "height": "72"},
        renderer="software",
    )
    nearest = (float("inf"), None)
    try:
        for seed in range(1000, 1600):
            lab.reset(seed=seed)
            start = pose(lab)
            distance = float(np.linalg.norm(start - target))
            if distance < nearest[0]:
                nearest = (distance, seed)
            if not (100 <= distance < 180):
                continue
            assert not (target[0] - 50 <= start[0] < target[0] + 50 and
                        target[1] - 50 <= start[1] < target[1] + 50), (site, seed, start)
            for _ in range(24):
                old_distance = float(np.linalg.norm(pose(lab) - target))
                for action in ACTIONS:
                    reward = lab.step(action, num_steps=4)
                    if reward > 0:
                        assert reward == 10, (site, seed, reward)
                        assert not lab.is_running(), (site, seed, "pickup did not end episode")
                        print(site, "PASS", "seed", seed, "start", start.tolist(), "reward", reward)
                        return
                    if not lab.is_running():
                        break
                    if np.linalg.norm(pose(lab) - target) < old_distance:
                        break
                if not lab.is_running():
                    break
        raise AssertionError((site, "no legal reward entry", "nearest_start", nearest))
    finally:
        lab.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runfiles", type=Path, required=True)
    args = parser.parse_args()
    deepmind_lab.set_runfiles_path(str(args.runfiles.resolve(strict=True)))
    for site in SITES:
        run_site(site)


if __name__ == "__main__":
    main()
