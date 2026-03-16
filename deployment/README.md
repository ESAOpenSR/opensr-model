# OpenSR HPC Launcher

This directory contains an installable Slurm launcher for running `opensr-model` over `cubo`-staged Sentinel-2 cutouts.

## What it does

- stages Sentinel-2 cutouts with `cubo`
- submits single-patch, grid, or AOI runs through `sbatch`
- runs `opensr-model` and `opensr-utils` on worker nodes
- writes manifests, logs, metadata, and outputs into one run directory

## Install

```bash
pip install -e .[hpc]
```

This exposes the `opensr-hpc` CLI.

## Example commands

```bash
opensr-hpc validate-config --config deployment/configs/runtime.default.yaml

opensr-hpc submit patch \
  --config deployment/configs/runtime.default.yaml \
  --lat 52.5200 \
  --lon 13.4050 \
  --start-date 2025-07-01 \
  --end-date 2025-07-03 \
  --dry-run

opensr-hpc submit grid \
  --config deployment/configs/runtime.default.yaml \
  --lat1 52.3 --lon1 12.9 \
  --lat2 52.7 --lon2 13.8 \
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

## Runtime configs

- `deployment/configs/runtime.default.yaml` - baseline config
- `deployment/configs/runtime.a100.example.yaml` - example GPU-node override
- set `aoi.path` in the runtime config to define a default shapefile AOI

## Run layout

```text
runs/<run_id>/
  run_manifest.yaml
  resolved_config.yaml
  submission/
  logs/
  patches/
    patch_000001/
      manifest.yaml
      inputs/lr.tif
      outputs/
      metadata/
```

## Notes

- the bundled Slurm entrypoint is `deployment/opensr_hpc/slurm/slurm_task_entrypoint.sh`
- default model config resolves to the packaged `opensr_model/configs/config_10m.yaml`
- set `model.checkpoint_path` in the runtime config if you want to pin a local checkpoint
- AOI submission accepts either a `.shp` file or a directory containing exactly one `.shp`; sidecar files like `.shx`, `.dbf`, and `.prj` must sit alongside it
