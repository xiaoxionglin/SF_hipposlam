"""Check the Python stack, ResNet-18, and a patched DMLab reset/step.

Run from an activated environment after following the README installation steps.
Use --cuda to require a working GPU, and --pretrained to check ImageNet weights
(downloads weights on the first run). This does not launch training.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Require and test CUDA")
    parser.add_argument("--pretrained", action="store_true", help="Load ImageNet ResNet-18 weights")
    args = parser.parse_args()

    import deepmind_lab
    import dm_env
    import numpy as np
    import torch
    import torchvision
    from torchvision.models import ResNet18_Weights, resnet18

    # Exercise imports used by the training runtime and the paper's encoder.
    runtime = importlib.import_module("sample_factory.train")
    importlib.import_module("sf_xxl.dmlab.custom_encoder")
    importlib.import_module("sf_xxl.dmlab.custom_core")
    importlib.import_module("sf_xxl.dmlab.custom_decoder")

    torch.set_num_threads(1)
    torch.manual_seed(0)
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("--cuda requested but CUDA is unavailable")
    device = torch.device("cuda:0" if args.cuda else "cpu")

    array = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.array_equal(torch.from_numpy(array).numpy(), array)
    x = torch.randn(32, 32, device=device, requires_grad=True)
    (x @ x.T).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()

    weights = ResNet18_Weights.DEFAULT if args.pretrained else None
    model = resnet18(weights=weights).eval().to(device)
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 72, 96, device=device))
    assert output.shape == (1, 1000) and torch.isfinite(output).all()
    if args.cuda:
        torch.cuda.synchronize(device)

    lab = deepmind_lab.Lab(
        "openfield_map2_fixed_loc3_fixedlength_noreward",
        ["RGB_INTERLEAVED"],
        config={"width": "96", "height": "72", "fps": "60"},
        renderer="software",
    )
    try:
        lab.reset(seed=99)
        frame = lab.observations()["RGB_INTERLEAVED"]
        assert frame.shape == (72, 96, 3) and frame.std() > 0
        action = np.zeros(len(lab.action_spec()), dtype=np.intc)
        reward = sum(float(lab.step(action, num_steps=8)) for _ in range(16))
        assert reward == 0 and lab.is_running()
        assert np.isfinite(lab.observations()["RGB_INTERLEAVED"]).all()
    finally:
        lab.close()

    print(
        json.dumps(
            {
                "status": "passed",
                "python": platform.python_version(),
                "executable": sys.executable,
                "numpy": np.__version__,
                "setuptools": version("setuptools"),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "dm_env": version("dm-env"),
                "deepmind_lab": version("deepmind-lab"),
                "runtime": runtime.__file__,
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if args.cuda else None,
                "pretrained_resnet18": args.pretrained,
                "dmlab_frames": 128,
                "reward": reward,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
