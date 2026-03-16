from __future__ import annotations

import logging
from pathlib import Path

from deployment.opensr_hpc.checkpoint import resolve_checkpoint_path, sha256sum
from deployment.opensr_hpc.config import InferenceConfig
from deployment.opensr_hpc.inference import run_inference
from deployment.opensr_hpc.manifests import read_yaml, write_json
from deployment.opensr_hpc.metadata import write_software_metadata
from deployment.opensr_hpc.raster import raster_validity_stats


LOGGER = logging.getLogger("opensr-hpc")


def _resolve_patch_manifest(manifest_path: Path, task_index: int | None) -> dict:
    manifest = read_yaml(manifest_path)
    if "tasks" not in manifest:
        return manifest
    if task_index is None:
        raise ValueError("Array task manifest requires task index")
    task_entries = manifest["tasks"]
    task = task_entries[task_index]
    return read_yaml((manifest_path.parent / Path(task["manifest"])).resolve())


def _resolve_manifest_local_path(manifest_path: Path, relative_path: str) -> Path:
    return (manifest_path.parent / relative_path).resolve()


def run_task(manifest_path: Path, task_index: int | None = None) -> Path | None:
    manifest_path = manifest_path.resolve()
    root_manifest = read_yaml(manifest_path)
    if "tasks" in root_manifest:
        if task_index is None:
            raise ValueError("Array task manifest requires task index")
        manifest_path = (manifest_path.parent / Path(root_manifest["tasks"][task_index]["manifest"])).resolve()
    manifest = _resolve_patch_manifest(manifest_path, None)
    config = manifest["config"]
    output_dir = _resolve_manifest_local_path(manifest_path, manifest["paths"]["output_dir"])
    metadata_dir = _resolve_manifest_local_path(manifest_path, manifest["paths"]["metadata_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    input_tif = _resolve_manifest_local_path(manifest_path, manifest["paths"]["input_tif"])
    patch_id = str(manifest.get("patch_id", "unknown"))

    LOGGER.info("starting worker task patch_id=%s manifest=%s", patch_id, manifest_path)
    LOGGER.info("checking staged raster patch_id=%s input_tif=%s", patch_id, input_tif)

    validity = raster_validity_stats(input_tif)
    if validity["valid_pixels"] == 0 or validity["nonzero_pixels"] == 0:
        LOGGER.info("skipping patch_id=%s because input raster is empty or all-zero stats=%s", patch_id, validity)
        write_json(
            metadata_dir / "result.json",
            {
                "status": "skipped",
                "reason": "empty_input_raster",
                "input_tif": str(input_tif),
                **validity,
            },
        )
        write_software_metadata(metadata_dir / "software_env.json")
        return None

    inference = InferenceConfig(
        factor=config["inference"]["factor"],
        window_size=tuple(config["inference"]["window_size"]),
        batch_size=config["inference"].get("batch_size", 2),
        overlap=config["inference"]["overlap"],
        eliminate_border_px=config["inference"]["eliminate_border_px"],
        gpus=config["inference"]["gpus"],
        save_preview=config["inference"]["save_preview"],
    )

    checkpoint_path = resolve_checkpoint_path(config["model"]["checkpoint_path"])
    LOGGER.info(
        "running inference patch_id=%s input_tif=%s checkpoint=%s",
        patch_id,
        input_tif,
        checkpoint_path if checkpoint_path is not None else "<default>",
    )
    final_output = run_inference(
        input_tif=input_tif,
        output_dir=output_dir,
        model_config_path=Path(config["model"]["config_path"]).expanduser().resolve(),
        checkpoint_path=checkpoint_path,
        inference=inference,
    )

    checkpoint_metadata = {}
    if checkpoint_path is not None:
        checkpoint_metadata = {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256sum(checkpoint_path),
        }

    write_json(
        metadata_dir / "result.json",
        {
            "final_output": str(final_output),
            **checkpoint_metadata,
        },
    )
    write_software_metadata(metadata_dir / "software_env.json")
    LOGGER.info("completed worker task patch_id=%s output=%s", patch_id, final_output)
    return final_output
