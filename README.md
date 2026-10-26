# Prediction and planning student starter

For flag-enabled training, start with the
[integrated DG world model](sf_xxl/dmlab/WORLD_MODEL.md): append
`--dg_world_model=true` to the existing HippoSLAM training command. The predictor
trains alongside PPO and can be edited directly under `sf_xxl/dmlab/`.

The [IntrMotiv prediction starter](exports/intrmotiv_prediction_starter/README.md)
contains reusable predictor components, a runnable example, focused tests, and
a Sample Factory integration guide for the action-input / world-model project.
It is also available as a [standalone ZIP](exports/intrmotiv_prediction_starter_v0.1.0.zip).

# Quick start

Use Linux and a fresh Python 3.10 environment.

## Prerequisites: rendering libraries

DMLab needs native rendering libraries; pip does not install these.
On Ubuntu, software rendering may require:

```bash
sudo apt-get install libsdl2-2.0-0 libosmesa6 libosmesa6-dev
```

On NEMO2, load the available modules before running DMLab:

```bash
module load lib/sdl2/2.28.2-gcccore-12.3.0 vis/mesa/24.1.3-gcccore-13.3.0
```

## 1. Fork and clone

[Fork this repository](https://github.com/xiaoxionglin/SF_hipposlam/fork), then clone
**your fork**. Replace `YOUR_GITHUB_USERNAME` with your GitHub username; use your
fork's repository name if you renamed it.

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/SF_hipposlam.git
cd SF_hipposlam
```

## 2. Create the environment

> **Mamba:** If you don't have `mamba`, install
> [Miniforge](https://github.com/conda-forge/miniforge#install), which includes it.
> Accept shell initialization when prompted, then reopen your terminal.

```bash
mamba create -n yourname python=3.10.12 pip
mamba activate yourname
```

## 3. Install Torch and TorchVision

Use the CUDA 12.8 build by default. It supports both CPU operations and GPU
training; using the GPU requires a compatible NVIDIA driver.

```bash
python -m pip install --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

<details>
<summary>Optional CPU-only build</summary>

For a smaller installation without CUDA dependencies, use this command
**instead of** the CUDA command above:

```bash
python -m pip install --no-deps torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

</details>

<details>
<summary>Other CUDA builds and version compatibility</summary>

The default pair was previously verified on G500 with Python 3.10 and CUDA 12.8
wheels. Choose another build compatible with your hardware and driver using
[PyTorch's installation instructions](https://pytorch.org/get-started/previous-versions/).
The requirements file accepts the selected CPU or CUDA build of the pinned pair.
Changing the Torch/TorchVision versions also requires updating their pins in
`requirements.txt`; see the [compatibility table](https://github.com/pytorch/vision#installation).
The pinned pair does not support Python 3.8.

</details>

## 4. Install the DMLab wheel

Download a compatible [DMLab wheel](https://drive.google.com/file/d/1YEXjm06f79KY5LB4NZ0e4xce3j1dcuxf/view?usp=drive_link)
and replace the path below with your download:

```bash
python -m pip install --no-deps /path/to/deepmind_lab-1.0-py3-none-any.whl
```

## 5. Install dependencies, patch, and verify

```bash
python -m pip install -r requirements.txt
bash deepmindlab_patch/patch_deepmindlab.sh
python -m pip check
python scripts/check_installation.py
```

<details>
<summary>What the requirements file installs</summary>

Run these commands from the root of your cloned fork.

The requirements file pins NumPy, setuptools, Torch, TorchVision, and `dm-env`,
then installs this checkout with `-e .`. The remaining runtime dependencies come
from `setup.py`; their full dependency tree is not locked.

The two `--no-deps` commands select platform-specific wheels first. The subsequent
requirements install resolves their dependencies together, including NumPy 1.26.4
and setuptools 65.5.0 for the existing DMLab wheel. No NumPy uninstall is needed
for this fresh-environment procedure. The environment is ready after completing
all steps, including the DMLab patch for custom maps and transparent rewards.

The linked wheel or a wheel provided by Sample Factory must match your system.
The patch script requires an active conda/mamba environment.

</details>

<details>
<summary>Optional environment and development extras</summary>

DMLab needs no optional environment extras. Install only those you use, for example:

```bash
python -m pip install -e ".[mujoco,atari,vizdoom]"
python -m pip check
```

The `dev` extra also installs additional environment and documentation dependencies.

</details>

<details>
<summary>Verification, pretrained ResNet-18, and CUDA checks</summary>

The check imports the training runtime and paper model modules, tests NumPy/Torch
conversion and gradients, runs ResNet-18 inference, and resets and steps a patched
custom DMLab map for 128 engine frames with software rendering. It does not launch
training.

To also load ImageNet weights (downloaded on first use):

```bash
python scripts/check_installation.py --pretrained
```

For a GPU installation, require CUDA and test the first visible GPU:

```bash
python scripts/check_installation.py --cuda --pretrained
```

The fresh-environment procedure was verified on 2026-10-26 with Python 3.10.12,
NumPy 1.26.4, setuptools 65.5.0, Torch 2.8.0+cpu, TorchVision 0.23.0+cpu, `dm-env`
1.6, and the existing DMLab 1.0 wheel. Dependency checks, runtime imports, gradients,
pretrained ResNet-18 inference, and the patched DMLab reset/step passed. CUDA was
not re-tested on this host because its NVIDIA driver was unavailable. The focused
tensor dictionary, action-space, and running-statistics tests also passed (13 tests).

</details>

# Troubleshooting

<details>
<summary>NumPy compatibility errors</summary>

Use the fresh-environment procedure above first. For an existing environment with
an error mentioning `numpy.ndarray` or NumPy's binary interface, check the
traceback and installed versions. To repair an incompatible NumPy installation:

```bash
python -m pip uninstall -y numpy
python -m pip install -r requirements.txt
python -m pip check
```

If the error persists, the affected compiled package may need a compatible wheel
or a rebuild; reinstalling NumPy alone does not fix every binary mismatch.

</details>

<details>
<summary>Missing pkg_resources</summary>

The existing DMLab wheel requires an older setuptools version, pinned by the
requirements file. To repair an environment where it was upgraded:

```bash
python -m pip install "setuptools==65.5.0"
```

</details>

<details>
<summary>TorchVision missing or incompatible</summary>

Reinstall the matched Torch/TorchVision pair using the selected CPU/CUDA command
above, then run `python -m pip install -r requirements.txt` and `python -m pip check`.

</details>

# Citation

This project is accepted at ICLR 2026

https://openreview.net/forum?id=li1vfqDzRD


code related to the paper is in the folder [sf_xxl](sf_xxl)
