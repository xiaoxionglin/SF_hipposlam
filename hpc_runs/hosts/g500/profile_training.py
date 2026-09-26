"""G500 host resource sampling shared by direct training queues.

The direct queue imports only :func:`resources`; keeping the host probe here
avoids duplicating admission logic in each study launcher.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import psutil


def resources():
    """Return one JSON-serializable host and GPU resource snapshot."""
    result = {
        "time": time.time(),
        "cpu_percent": psutil.cpu_percent(),
        "load": os.getloadavg(),
        "available_ram_gib": psutil.virtual_memory().available / 2**30,
    }
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    result["gpus"] = [
        dict(
            zip(
                ("index", "free_mib", "utilization", "power_w"),
                map(float, row.split(",")),
            )
        )
        for row in query.stdout.strip().splitlines()
    ]
    result["pressure"] = {name: Path(f"/proc/pressure/{name}").read_text().strip() for name in ("cpu", "memory", "io")}
    return result
