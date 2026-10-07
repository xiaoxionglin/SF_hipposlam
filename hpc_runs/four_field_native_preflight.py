"""Compute-node DMLab check for the four prescribed DG fields."""

import json
import os
from pathlib import Path

import deepmind_lab
import numpy as np

from sf_working_directories.IntrMotiv.dmlab.dmlab_gym import NAVIGATION_ACTION_SET
from sf_working_directories.IntrMotiv.dmlab.prescribed_dg import (
    FIELD_CENTERS,
    FIELD_RADIUS,
    prescribed_activity,
)


runfiles = os.environ.get("DMLAB_RUNFILES")
if runfiles:
    deepmind_lab.set_runfiles_path(str(Path(runfiles).resolve(strict=True)))

lab = deepmind_lab.Lab(
    "openfield_map2_fixed_loc3_fixedlength_noreward",
    ["DEBUG.POS.TRANS", "DEBUG.POS.ROT"],
    config={"width": "96", "height": "72"},
    renderer="software",
)
actions = [np.asarray(a, dtype=np.intc) for a in NAVIGATION_ACTION_SET]
rng = np.random.default_rng(20261007)
counts = np.zeros(4, dtype=np.int64)
entries = np.zeros(4, dtype=np.int64)
nearest = np.full(4, np.inf)
samples = 0
resets = 0
prior = np.zeros(4, dtype=bool)


def observe():
    global samples, prior
    position = np.asarray(lab.observations()["DEBUG.POS.TRANS"], dtype=float)
    distance = np.linalg.norm(FIELD_CENTERS - position[:2], axis=1)
    nearest[:] = np.minimum(nearest, distance)
    active = prescribed_activity(position, "gaussian4") > 0
    counts[:] += active
    entries[:] += active & ~prior
    prior = active
    samples += 1


try:
    for seed in range(1000, 1200):
        lab.reset(seed=seed)
        prior[:] = False
        resets += 1
        observe()
    lab.reset(seed=20261007)
    prior[:] = False
    observe()
    for _ in range(12000):
        if not lab.is_running():
            resets += 1
            lab.reset(seed=20261007 + resets)
            prior[:] = False
            observe()
        lab.step(actions[int(rng.integers(len(actions)))], num_steps=8)
        if lab.is_running():
            observe()
finally:
    lab.close()

print(json.dumps({
    "samples": samples,
    "resets": resets,
    "field_radius": FIELD_RADIUS,
    "counts": counts.tolist(),
    "entries": entries.tolist(),
    "nearest_distance": nearest.tolist(),
}, indent=2))
