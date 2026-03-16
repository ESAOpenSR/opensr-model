from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import OmegaConf

from deployment.opensr_hpc.config import InferenceConfig
from deployment.opensr_hpc.naming import patch_output_name
from deployment.opensr_hpc.raster import compress_geotiff, compute_centroid_lat_lon


LOGGER = logging.getLogger("opensr-hpc")


def _build_runner(opensr_utils: Any, inference: InferenceConfig):
    class ConfigurableLargeFileProcessing(opensr_utils.large_file_processing):
        def __init__(self, *args: Any, batch_size: int, **kwargs: Any) -> None:
            self._opensr_hpc_batch_size = batch_size
            super().__init__(*args, **kwargs)

        def create_datamodule(self) -> None:
            from opensr_utils.data_utils.datamodule import PredictionDataModule

            dm = PredictionDataModule(
                input_type=self.input_type,
                root=self.root,
                windows=self.image_meta["image_windows"],
                lr_file_dict=self.image_meta["lr_file_dict"],
                prefetch_factor=2,
                batch_size=self._opensr_hpc_batch_size,
                num_workers=4,
            )
            dm.setup()
            self._log(
                f"📦 Created PredictionDataModule with {len(dm.dataset)} patches (batch_size={self._opensr_hpc_batch_size})."
            )
            self.datamodule = dm

    return ConfigurableLargeFileProcessing


def run_inference(
    *,
    input_tif: Path,
    output_dir: Path,
    model_config_path: Path,
    checkpoint_path: Path | None,
    inference: InferenceConfig,
) -> Path:
    try:
        import opensr_model
        import opensr_utils
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "opensr-hpc inference requires optional dependencies. Install with `pip install \"opensr-model[hpc]\"`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info(
        "loading inference model input_tif=%s device=%s window_size=%s batch_size=%s",
        input_tif,
        device,
        inference.window_size,
        inference.batch_size,
    )
    cfg = OmegaConf.load(model_config_path)
    if checkpoint_path is not None:
        cfg.ckpt_version = str(checkpoint_path)

    model = opensr_model.SRLatentDiffusion(cfg, device=device)
    model.load_pretrained(cfg.ckpt_version)

    runner_cls = _build_runner(opensr_utils, inference)
    runner = runner_cls(
        root=str(input_tif),
        model=model,
        window_size=tuple(inference.window_size),
        batch_size=inference.batch_size,
        factor=inference.factor,
        overlap=inference.overlap,
        eliminate_border_px=inference.eliminate_border_px,
        device=device,
        gpus=cast(Any, inference.gpus),
        save_preview=inference.save_preview,
        debug=False,
    )
    LOGGER.info("inference finished for input_tif=%s", input_tif)

    final_sr_path = getattr(runner, "final_sr_path", None)
    if final_sr_path is None:
        final_sr_path = output_dir / "sr.tif"
    else:
        final_sr_path = Path(final_sr_path)

    if not final_sr_path.exists():
        raise FileNotFoundError(f"Expected SR output at {final_sr_path}")

    latitude, longitude = compute_centroid_lat_lon(final_sr_path)
    compressed_path = output_dir / patch_output_name(latitude, longitude)
    LOGGER.info("compressing SR GeoTIFF input=%s output=%s", final_sr_path, compressed_path)
    compress_geotiff(final_sr_path, compressed_path)
    final_sr_path.unlink(missing_ok=True)
    return compressed_path
