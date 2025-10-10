#!/usr/bin/env python3
"""
demo.py — End-to-end example: using `opensr-model` together with `opensr-utils`
to super-resolve a full Sentinel-2 tile.

What this shows (without changing your actual code):
  • How to instantiate the SR model from `opensr-model` using a YAML config.
  • How to run large-file, patch-based inference with `opensr-utils.large_file_processing`.
  • What each key parameter does in practice (windowing, overlap, device routing, etc.).

Prereqs
  • Python ≥3.10 recommended.
  • CUDA GPU (for speed), recent PyTorch build with CUDA.
  • `opensr-model` installed and its checkpoints accessible (see `ckpt_version` in YAML).
  • `opensr-utils` > 1.0.0 (this demo uses the newer large-file pipeline).
  • Disk space: SR outputs for full tiles can be large (hundreds of MB to multi-GB).

Notes
  • This example uses a Sentinel-2 `.SAFE` folder as input. `opensr-utils` also supports
    other sources (e.g., single GeoTIFFs, S2GM folders) via the same `root` argument.
  • The script pins inference to one GPU via CUDA_VISIBLE_DEVICES; adjust to your setup.
  • `window_size` is specified in **LR pixels**. With `factor=4`, 128→512 on output.
  • `overlap` softly blends neighboring patches to suppress seam artifacts.
  • `eliminate_border_px` trims the per-patch border before stitching (helps kill halos).
  • `large_file_processing` handles tiling, streaming IO, stitching, and saving.
"""

# This script is an example how opensr-utils can be used in unison with opensr-model
# in order to SR a while S2 tile.
# This requires opensr-utils > 1.0.0

# Imports
import opensr_model
import torch,os
from omegaconf import OmegaConf
os.environ["CUDA_VISIBLE_DEVICES"]="0" # run only on GPU 0

# Import and Instanciate SR Model
device = "cuda" if torch.cuda.is_available() else "cpu" # Select GPU device
config = OmegaConf.load("opensr_model/configs/config_10m.yaml") # load config
model = opensr_model.SRLatentDiffusion(config, device=device) # create model
model.load_pretrained(config.ckpt_version) # load checkpoint

import opensr_utils
print("Using opensr-utils version:", opensr_utils.__version__)

# Define path to S2 tile
path = "/path/to/folder/S2A_MSIL2A_XXXXXXXXTXXXXXX_XXXXX_XXXX_XXXXXX_XXXXXXXXTXXXXXX.SAFE/"
# The `.SAFE` folder must contain the expected Sentinel-2 structure (GRANULE/, IMG_DATA/,
# metadata XMLs, etc.). `opensr-utils` will resolve band paths, handle georeferencing,
# and stream windows directly from disk to avoid loading the whole tile at once.
# You can also pass a single GeoTIFF or a folder of GeoTIFFs here (e.g., S2GM data).

# Create and run SR on .SAFE S2 format
sr_object = opensr_utils.large_file_processing(
    root=path,                 # File or Folder path (SAFE, S2GM, or single GeoTIFF)
    model=model,               # your SR model (forward(x) -> SR), already on `device`
    window_size=(128, 128),    # LR window size for patching; 128 @ x4 → 512px HR tiles
    factor=4,                  # SR factor (10m → 2.5m); used to scale & stitch outputs
    overlap=12,                # LR overlap for seam-free blending (x4 → 48px in SR space)
    eliminate_border_px=2,     # trim LR-space borders per patch before blending
    device=device,             # "cuda" or "cpu"; `opensr-utils` dispatches tensors here
    gpus=0,                    # pass a GPU ID (int) or list of IDs for multi-GPU setups
)
# `sr_object` exposes:
#   • paths to intermediate & final outputs (e.g., stitched SR GeoTIFF/COG),
#   • logs/metadata about tiling & timing,
#   • helper accessors for debugging. See opensr-utils docs for the exact API.
#
# Tuning pointers:
#   • Increase `window_size` if you have more VRAM (depends on model requirements).
#   • Adjust `overlap` if you see seams; typical LR values are 8–24 for x4 SR.
#   • If border artifacts remain, a slightly larger `eliminate_border_px` can help.
#   • For multi-GPU, pass e.g. `gpus=[0,1]`, your model will be multi-GPU optimized automatically
