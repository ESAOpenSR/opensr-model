# Development beyond the paper
This repository continues development beyond the paper. The [`paper-release`](https://github.com/ESAOpenSR/opensr-model/tree/paper-release) branch preserves the previous `main` snapshot (v1.1.1).

**v1.2.0** adds DPM-Solver++ sampling, two-stage training, and PyTorch spectral
back-projection. Inference now defaults to **3-step DPM-Solver++ with
back-projection**. See the [release notes and migration example](docs/changelog.md#v120--2026-10-12).

<img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/opensr_logo.png?raw=true" width="250"/>

# Latent Diffusion Super-Resolution - Sentinel 2 (LDSR-S2)
<p align="center">
  <img src="https://github.com/ESAOpenSR/opensr-model/blob/main/resources/ldsr-s2_schema.png?raw=true" alt="S2 example" width="75%">
</p>

**Description**: This repository contains the code of the paper [Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery with Latent Diffusion](https://ieeexplore.ieee.org/abstract/document/10887321). This package contains the latent-diffusion model and weights to super-resolute the RGB-NIR bands of Sentinel-2. It can be embedded in the "opensr-utils" package in order to create a data pipeline for SR products. 

**🧪 Status**: LDSR-S2 has exited the experimental phase as of **v1.0.0**  
📌 For super-resolving **20m bands**, check out [`SEN2SR`](https://github.com/ESAOpenSR/SEN2SR), or use it alongside LDSR-S2 in the third notebook linked in the SEN2SR README.
  
---

# Table of Contents
- [🚀 1. Interactive Notebooks - Google Colab](#1--interactive-notebooks---google-colab)
- [🛠️ 2. Install and Usage - Local](#2-install-and-usage---local)
  - [🧱 2.1 Weights and Checkpoints](#21-weights-and-checkpoints)
  - [🖥️ 2.2 HPC and Slurm Launcher](#22-hpc-and-slurm-launcher)
  - [2.3 Development, tests, and coverage](#23-development-tests-and-coverage)
- [🖼️ 3. Super-Resolution Examples](#3-super-resolution-examples)
  - [🛰️ 3.1 Example SR Image](#31-example-sr-image)
  - [🔎 3.2 Demo File and Uncertainty Calculation](#32-demo-file-and-uncertainty-calculation)
- [🗺️ 4. Inference on S2 Tiles: Data Pipeline](#4-inference-on-s2-tiles-data-pipeline)
- [📝 5. Notes](#5-notes)
  - [🎓 5.1 Training](#51-training)
  - [📚 5.2 Citation](#52-citation)
  - [📈 5.3 Status](#53-status)



## 1. 🚀 Interactive Notebooks - Google Colab

Run LDSR-S2 directly in Google Colab within seconds! These notebooks let you fetch Sentinel-2 imagery, apply super-resolution, and save results — with or without going through the code. The no-code version even implements a user interface to select and process data.

| Notebook Name                  | Description                                                                                      | Link                                                                 |
|-------------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------------------------|
| **LDSR-S2 No-Code**           | 🔘 No coding required — choose a point on a map and download SR results as GeoTIFFs               | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xhlVjGkHPF1znafSGrWtyZ0wzcogVRCe?usp=sharing) |
| **LDSR-S2 Walkthrough**              | 🧪 Code-level walkthrough with uncertainty estimation and advanced plotting                       | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1onza61SP5IyiUQM85PtyVLz6JovvQ0TN?usp=sharing) |
| **LDSR-S2 & SEN2SR**          | 🔄 Use LDSR-S2 alongside SEN2SR to compare results on 10m + 20m bands                             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NJuyswsquOLMFc_AP93P_5QcZnbNhGuB?usp=sharing) |

---



## 2. 🛠️ Install and Usage - Local
```bash
pip install opensr-model
```

Minimal Example  
```python
# Get Config
from io import StringIO
import requests
from omegaconf import OmegaConf
config_url = "https://raw.githubusercontent.com/ESAOpenSR/opensr-model/refs/heads/main/opensr_model/configs/config_10m.yaml"
response = requests.get(config_url)
config = OmegaConf.load(StringIO(response.text))

# Get Model
import torch
device = "cuda" if torch.cuda.is_available() else "cpu" # set device
import opensr_model # import package
model = opensr_model.SRLatentDiffusion(config, device=device) # create model
model.load_pretrained(config.ckpt_version) # load checkpoint
sr = model(torch.rand(1,4,128,128)) # DPM-Solver++ (3 steps) + back-projection
```  

**Inference defaults to DPM-Solver++ 2M with 3 steps and PyTorch back-projection.**
Diffusers is included in the standard package dependencies:

```bash
pip install -e .
```

```python
torch.manual_seed(42)
sr = model(image)  # DPM-Solver++ (3 steps) + back-projection
sr_more_steps = model(image, sampling_steps=5)
sr_legacy = model(image, sampler="ddim", sampling_steps=100, spectral_correction="histogram")
```

Set `denoiser_settings.sampler` in the model YAML to choose a default sampler.
With `sampling_steps: null`, DPM-Solver++ uses 3 steps and DDIM uses 100.
Call arguments override config values; apply runtime settings after
`load_pretrained`, which may restore checkpoint settings. The 3-step setting
prioritizes speed; output quality can vary by scene.

Spectral correction can be selected independently of the sampler:

```python
sr = model(image)  # back-projection is the default
sr_explicit = model(image, sampler="dpm_solver++", spectral_correction="back_projection")
sr_hist = model(image, spectral_correction="histogram")
sr_raw = model(image, spectral_correction="none")
```

Back-projection runs in PyTorch on decoded reflectances and adjusts the output so
that blur/downsampling reproduces the LR measurement. Its default operator is
4×4 block averaging; optional per-band Gaussian blur and stopping settings are
available through `back_projection={...}` or the model YAML. Configs without
`spectral_correction` also use back-projection. The new path keeps
nodata at zero and does not clamp corrected values. The legacy
`histogram_matching=False` still disables correction unless an explicit
`spectral_correction` is supplied.

See the [inference changelog](docs/changelog.md) for settings and compatibility,
and the [training guide](docs/training.md) for training instructions.

### 2.1 🧱 Weights and Checkpoints
The model should load automatically with the `model.load_pretrained` command. Alternatively, the checkpoints can be found on [HuggingFace](https://huggingface.co/simon-donike/RS-SR-LTDF/tree/main).

### 2.2 🖥️ HPC and Slurm Launcher

For cluster workflows, this repository now also ships an installable HPC launcher built around `cubo`, `opensr-model`, `opensr-utils`, and Slurm.

Install the optional extras:

```bash
pip install -e .[hpc]
```

This exposes the `opensr-hpc` CLI:

```bash
opensr-hpc validate-config --config deployment/configs/runtime.default.yaml

opensr-hpc submit patch \
  --config deployment/configs/runtime.default.yaml \
  --lat 52.5200 \
  --lon 13.4050 \
  --start-date 2025-07-01 \
  --end-date 2025-07-03 \
  --dry-run

opensr-hpc submit aoi \
  --config deployment/configs/runtime.default.yaml \
  --aoi-path /data/berlin_aoi \
  --start-date 2025-07-01 \
  --end-date 2025-07-03 \
  --dry-run
```

See `deployment/README.md` for the full run layout, config schema, AOI settings, and Slurm entrypoint details.

### 2.3 Development, tests, and coverage

Use Python 3.12 or newer and install the training dependencies to run the full
test suite. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[train,dev]'
python -m pre_commit install
python -m pre_commit run --all-files --hook-stage pre-push
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. Keep the environment active when pushing: both checks use its
Python and installed dependencies. Hook installation is required once per clone.

Each push checks Black formatting on the Python files included in the push and
runs the full pytest suite with line and branch coverage, including the distributed
tests. Formatting or test failures block the push. Fix formatting with
`python -m black .`, commit the changes, and push again. Commits do not run these
checks. To migrate an existing installation:

```bash
python -m pre_commit uninstall --hook-type pre-commit
python -m pre_commit install
```

GitHub Actions runs these checks when a GitHub release is published, and PyPI
publishing waits for them to pass. The GitHub release itself is already published
when these checks start. Ordinary pushes and pull requests do not run CI checks.
You can also run the Code quality and tests workflow manually from the Actions tab.
CI saves the coverage reports as a downloadable artifact.

Useful manual commands:

```bash
python -m black .                       # Format Python files
python -m black --check .               # Check formatting without editing
python -m pytest                       # Full suite and coverage reports
python -m pytest tests/diffusion --no-cov  # Focused tests without coverage
```

Coverage includes `opensr_model`, including modules the tests do not import.
The external HPC tooling in `deployment/` is outside the coverage scope.
The terminal shows missing lines; open `htmlcov/index.html` for
the browsable report or use `coverage.xml` with coverage tools. Reports are
Git-ignored. Coverage is reported without an initial minimum percentage; set
`fail_under` in `[tool.coverage.report]` in `pyproject.toml` to enforce a target.

Tests use small CPU models and limit PyTorch threads to keep push checks
practical. The CUDA comparison and released-checkpoint test skip when their
resources are unavailable. Git-ignored `tests/diagnostics` experiments are
excluded from the shared suite. The `test` extra includes pytest and coverage
tools; the `dev` extra also includes the pinned Black formatter and pre-commit.


# 3. 🖼️ Super-Resolution Examples

## 3.1 🛰️ Example SR Image
<p align="center">
  <img src="resources/example2.png" alt="S2 example" width="80%">
</p>

## 3.2 🔎 Demo File and Uncertainty Calculation
Run the `demo.py` file to see how the package works. It super-resolves an example tensor and saves the corresponding uncertainty map.

<p align="center">
  <img src="resources/sr_example.png" alt="SR example" width="80%">
</p>

<p align="center">
  <img src="resources/uncertainty_map.png" alt="Uncertainty map" width="50%">
</p>


# 4. 🗺️ Inference on S2 Tiles: Data Pipeline

This repository itself exposes the **raw model** for tensor‑level inference. By default it performs **128→512** super‑resolution on 4‑channel tensors (B×4×128×128 → B×4×512×512). It **does not** handle geospatial IO, tiling, or metadata.

For real‑world workflows on Sentinel‑2 data, please use **`opensr-utils`**:
- Run inference over **4‑band GeoTIFFs** (e.g., RGB‑NIR) or full Sentinel‑2 **`.SAFE` folders** (including **`.SAFE.zip`**).
- Handles **tiling**, **overlap blending**, **stitching**, and preserves **CRS/transform**.
- Works with every model (including this one), define LR‑space `window_size=128`, `factor=4`, overlap, and border trimming.
- Writes georeferenced outputs (e.g., GeoTIFF/COG), plus basic logs/metadata.

Minimal example (pseudocode; see `opensr-utils` for details):
```python
import opensr_utils
sr_job = opensr_utils.large_file_processing(
    root="/path/to/S2_tile.SAFE",  # also supports GeoTIFFs and .SAFE.zip
    model=model,                   # your SRLatentDiffusion instance
    window_size=(128, 128),        # LR patch size → 512px SR at factor=4
    factor=4,                      # 10m → 2.5m
    overlap=12,                    # LR‑space overlap to reduce seams
    eliminate_border_px=2,         # trim LR‑space border per tile
    device=device,                 # "cuda"/"cpu"
    gpus=0,                        # single GPU id or list for multi‑GPU
)
```

Use the notebooks and `demo.py` for quick starts; for full‑tile processing and proper georeferencing, prefer `opensr-utils`.



# 5. 📝 Notes

## 5.1 🎓 Training
The autoencoder and conditional diffusion model can be trained independently with PyTorch Lightning using paired TACO data. Runs save checkpoints for both training resume and inference.

See the [training guide](docs/training.md) and the packaged configurations in
[`opensr_model/configs/training_configs`](opensr_model/configs/training_configs).

## 5.2 📚 Citation
If you use this model in your work, please cite  
```tex
@ARTICLE{ldsrs2,
  author={Donike, Simon and Aybar, Cesar and Gómez-Chova, Luis and Kalaitzis, Freddie},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Trustworthy Super-Resolution of Multispectral Sentinel-2 Imagery With Latent Diffusion}, 
  year={2025},
  volume={18},
  number={},
  pages={6940-6952},
  doi={10.1109/JSTARS.2025.3542220}}
```

## 5.3 📈 Status
This repository has left the experimental stage with the publication of v1.0.0.   
[![PyPI Downloads](https://static.pepy.tech/badge/opensr-model)](https://pepy.tech/projects/opensr-model)
