from __future__ import annotations

import json
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from opensr_model.autoencoder.utils import AttnBlock
from opensr_model.training.runtime.config import load_training_config, package_config_path
from opensr_model.training.data.taco import InvalidPairError, OpenSRDataModule, TacoPairedDataset, _group_key
from opensr_model.training.modules.diffusion import DiffusionTrainingModule
from opensr_model.training.runtime.checkpoints import load_checkpoint_payload, load_diffusion_weights
from opensr_model.training.validation.metrics import guardrail_decision, paired_report
from opensr_model.training.validation.callback import TransferValidation, validation_inference
from opensr_model.training.validation.data import (
    digest_file, pad_lr, prepare_transfer, read_neon_pair, resolve_sen2neon, select_diverse,
    overlapping_groups,
)


def test_guardrails_exact_limits_and_nonfinite():
    baseline = {"rgb_lpips": .5, "nir_mae": .1, "sam": .2}
    assert guardrail_decision(dict(rgb_lpips=.4, nir_mae=.11, sam=.22), baseline)[0]
    assert not guardrail_decision(dict(rgb_lpips=.4, nir_mae=.11001, sam=.22), baseline)[0]
    assert not guardrail_decision(dict(rgb_lpips=.4, nir_mae=.11, sam=.22001), baseline)[0]
    for metric in baseline:
        for invalid in (float("nan"), float("inf"), None):
            candidate = dict(baseline, **{metric: invalid})
            assert not guardrail_decision(candidate, baseline)[0]


def test_preset_is_vanilla_and_paths_portable():
    config = load_training_config(package_config_path("train_sen2naip.yaml"))
    assert config.training.optimizer.learning_rate == 1e-5
    assert config.training.scheduler.name == "constant"
    for name in ("p2_gamma", "latent_detail_weight", "decoded_reconstruction_weight",
                 "decoded_detail_weight", "decoded_perceptual_weight"):
        assert config.training.sharpness[name] == 0
    assert config.data.batch_size * config.trainer.accumulate_grad_batches * config.trainer.devices == 8
    assert config.data.synthetic_hr_nir_dir is None
    assert config.training.validation.every_n_train_steps == 1000
    assert config.training.validation.sen2neon.preview_count == 8
    assert not config.training.validation.sen2neon.save_all_reconstructions
    assert config.logging.mode == "offline"


def test_packaged_training_paths_stay_relative_to_launch_directory(tmp_path, monkeypatch):
    from opensr_model.training.runtime.config import load_model_config
    paths = [package_config_path(f"train_{stage}.yaml") for stage in ("autoencoder", "diffusion", "sen2naip")]
    assert all(path.parent.name == "training_configs" and path.is_file() for path in paths)
    monkeypatch.chdir(tmp_path)
    for path in paths:
        config = load_training_config(path, overrides=["run_dir=my_run", "data.taco_path=data/pairs"])
        assert config.run_dir == str(tmp_path / "my_run")
        assert config.data.taco_path == str(tmp_path / "data/pairs")
        assert load_model_config(config.model.architecture_config).first_stage_config


def test_reconstruction_retention_preserves_scoring_and_final_reuse(tmp_path, monkeypatch, tiny_model_config):
    import lpips
    from opensr_model.training.validation import callback as callbacks
    rows = [{"id": str(i), "site": str(i), "landcover": "Forest"} for i in range(3)]
    prepared = {"request_hash": "fixed", "sen2neon_root": "/unused", "sen2neon_rows": rows,
                "selection_ids": [r["id"] for r in rows], "raw_rows": []}
    cb = TransferValidation(prepared, {"sen2neon": {"preview_count": 1, "save_all_reconstructions": False}}, tmp_path)
    trainer = types.SimpleNamespace(global_rank=0, world_size=1, global_step=0, is_global_zero=True,
                                    logger=None, strategy=types.SimpleNamespace(
                                        broadcast=lambda value, src: value, barrier=lambda: None))
    sample = {"LR_image": torch.ones(4, 8, 8) * .2, "image": torch.ones(4, 32, 32) * .2,
              "valid_mask": torch.ones(1, 32, 32, dtype=torch.bool), "clipped_fraction": 0.}
    monkeypatch.setattr(callbacks, "SEN2NEONDataset", lambda root, records: [sample for _ in records])
    monkeypatch.setattr(lpips, "LPIPS", lambda **kwargs: torch.nn.Identity())
    monkeypatch.setattr(callbacks, "reconstruction_scores", lambda sr, *args: {"rgb_lpips": float(sr.mean())})
    calls = []
    def sample_sr(trainer, module, lr, sample_id):
        calls.append(sample_id)
        return torch.ones(1, 4, 32, 32) * (int(sample_id) + 1) / 10
    cb._sample = sample_sr
    module = DiffusionTrainingModule(tiny_model_config, validation_num_samples=0)
    first = cb._evaluate(trainer, module, rows, "selection")
    assert len(calls) == len(first) == 3
    assert sum(r["file"] is not None for r in first) == 1
    assert len(list((tmp_path / "validation/selection").glob("*.npz"))) == 1
    assert len(json.loads((tmp_path / "validation/selection/metrics.json").read_text())) == 3
    reused = cb._evaluate(trainer, module, rows, "reused", reuse="selection")
    assert reused == first
    assert len(calls) == 3
    final = cb._evaluate(trainer, module, rows, "final", reuse="selection", save_all=True)
    # Only the two unretained predictions need to be sampled again.
    assert len(calls) == 5
    assert all(r["file"] for r in final)
    assert [r["metrics"] for r in final] == [r["metrics"] for r in first]
    assert len(list((tmp_path / "validation/final").glob("*.npz"))) == 3


def test_strict_filter_rejects_full_pair_before_crop(fake_taco_factory):
    bundle = fake_taco_factory(schema="crosssensor", include_nodata=False)
    # Fixture has one valid-as-reflectance zero at the far LR corner.
    ordinary = TacoPairedDataset(bundle.root, training=True, train_patch_size=16)
    ordinary.validate_sample()
    strict = TacoPairedDataset(bundle.root, training=True, train_patch_size=16, strict_full_validity=True)
    with pytest.raises(InvalidPairError):
        strict.validate_sample()
    with rasterio.open(bundle.raster_paths["lr"], "r+") as dst:
        a = dst.read()
        a[a == 0] = 1
        dst.write(a)
    strict.validate_sample()
    assert _group_key({}, "NA5120_E1234N5678__different_20200101") == _group_key({}, "NA5120_E1234N5678__other_20220202")


def test_preparation_cache_and_source_change(tmp_path, fake_taco_factory):
    bundle = fake_taco_factory(schema="crosssensor", include_nodata=False)
    with rasterio.open(bundle.raster_paths["lr"], "r+") as dst:
        a = dst.read()
        a[a == 0] = 1
        dst.write(a)
    config = load_training_config(package_config_path("train_sen2naip.yaml"), overrides=[
        f"data.taco_path={bundle.root}", "data.train_patch_size=16",
        "training.validation.sen2neon.enabled=false", "data.validation_patch_size=16",
    ])
    run = tmp_path / "run"
    a = prepare_transfer(config, run)
    assert len(a["train_ids"]) + len(a["val_ids"]) == 12
    assert not a["excluded"]
    assert not set(a["train_ids"]) & set(a["val_ids"])
    config.data.taco_path = [str(p) for p in bundle.taco_paths]
    b = prepare_transfer(config, run)
    assert a == b
    config.data.split_seed += 1
    with pytest.raises(ValueError, match="changed"):
        prepare_transfer(config, run)


def test_optional_dataset_missing_and_broken(tmp_path):
    missing = tmp_path / "absent"
    assert resolve_sen2neon({"root": str(missing)}) == (None, [], None)
    with pytest.raises(FileNotFoundError):
        resolve_sen2neon({"enabled": True, "root": str(missing)})
    missing.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_sen2neon({"root": str(missing)})


def test_packaged_subset_covers_sites_and_developed():
    rows = json.loads(package_config_path("sen2neon_clean192.json").read_text())
    subset = select_diverse(rows, 48)
    assert len(subset) == 48
    assert {r["site"] for r in subset} == {r["site"] for r in rows}
    assert {r["id"] for r in rows if r["landcover"] == "Developed"} <= {r["id"] for r in subset}
    assert subset == select_diverse(list(reversed(rows)), 48)


def test_neon_full_grid_band_order_and_zero_rejection(tmp_path):
    row = {"id": "example", "lr": "s2_l2a_10m/example.tif", "hr": "neon_2.5m_linearized/example.tif"}
    for key, size, scale, nodata in (("lr", 256, 10, 65535), ("hr", 1024, 2.5, 0)):
        path = tmp_path / row[key]
        path.parent.mkdir()
        data = np.broadcast_to(np.arange(1, 13, dtype=np.uint16)[:, None, None] * 100, (12, size, size))
        with rasterio.open(path, "w", driver="GTiff", width=size, height=size, count=12,
                           dtype="uint16", crs="EPSG:32632", transform=from_origin(500000, 4500000, scale, scale),
                           nodata=nodata, compress="deflate") as dst:
            dst.write(data)
    sample = read_neon_pair(tmp_path, row)
    assert torch.allclose(sample["LR_image"][:, 0, 0], torch.tensor([.04, .03, .02, .08]))
    assert sample["image"].shape == (4, 1024, 1024)
    with rasterio.open(tmp_path / row["hr"], "r+") as dst:
        dst.write(np.zeros((1, 1), dtype=np.uint16), 8, window=rasterio.windows.Window(0, 0, 1, 1))
    with pytest.raises(InvalidPairError):
        read_neon_pair(tmp_path, row)


def test_attention_exact_and_context_restores_rng(tiny_model_config):
    block = AttnBlock(32).eval()
    x = torch.randn(1, 32, 9, 7)
    original = block(x)
    block.query_chunk_size = 11
    assert torch.allclose(block(x), original, atol=2e-6, rtol=2e-5)
    module = DiffusionTrainingModule(tiny_model_config, validation_num_samples=0)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError):
        with validation_inference(module, 7):
            assert not module.training
            torch.rand(7)
            raise RuntimeError("sampling failure")
    assert module.training
    assert torch.equal(before, torch.random.get_rng_state())
    assert all(m.query_chunk_size is None for m in module.first_stage_model.modules() if isinstance(m, AttnBlock))


def test_padding_preserves_observed_pixels():
    lr = torch.rand(1, 4, 121, 130)
    padded, shape = pad_lr(lr)
    assert padded.shape[-2:] == (128, 136)
    assert shape == (484, 520)
    assert torch.equal(padded[..., :121, :130], lr)


def test_alignment_roundoff_and_geographic_group_exclusion(monkeypatch):
    from affine import Affine
    from types import SimpleNamespace
    import pandas as pd
    from opensr_model.training.data.taco import _RasterAsset, _validate_pair_alignment, TacoSampleRecord
    lr_transform = from_origin(500000, 4500000, 10, 10)
    hr_transform = from_origin(500000, 4500000.00012, 2.5, 2.5)
    asset = lambda transform, shape: _RasterAsset(torch.ones(4, *shape), torch.ones(1, *shape, dtype=torch.bool), shape, "EPSG:32632", transform)
    _validate_pair_alignment(asset(lr_transform, (8, 8)), asset(hr_transform, (32, 32)), sample_id="x", factor=4)
    with pytest.raises(InvalidPairError):
        _validate_pair_alignment(asset(lr_transform, (8, 8)), asset(Affine.translation(1, 0) * hr_transform, (32, 32)), sample_id="x", factor=4)
    table = pd.DataFrame([
        {"stac:geotransform": (0, .01, 0, 1, 0, -.01), "stac:raster_shape": (10, 10), "stac:crs": "EPSG:4326"},
        {"stac:geotransform": (30, .01, 0, 30, 0, -.01), "stac:raster_shape": (10, 10), "stac:crs": "EPSG:4326"},
    ])
    monkeypatch.setattr("opensr_model.training.validation.data._tacoreader_v1", lambda: SimpleNamespace(load=lambda p: table))
    records = [TacoSampleRecord(Path("x"), 0, "a", "same-location"), TacoSampleRecord(Path("x"), 1, "b", "same-location")]
    assert overlapping_groups(records, [[.02, .92, .03, .93]]) == {"same-location"}


def test_site_bootstrap_uses_paired_tiles():
    a = [{"id": str(i), "site": str(i // 2), "metrics": {"mae": float(i)}} for i in range(6)]
    b = [{"id": r["id"], "site": r["site"], "metrics": {"mae": r["metrics"]["mae"] + 2}} for r in a]
    report = paired_report(a, b, resamples=50)
    assert report["metrics"]["mae"]["mean_delta"] == 2
    assert report["metrics"]["mae"]["ci95"] == [2, 2]


def test_selection_exports_ema_and_resumes_original_baseline(tmp_path, tiny_model_config, fake_taco_factory):
    bundle = fake_taco_factory(hr_size=16, include_nodata=False)
    prepared = {"request_hash": "fixed", "sen2neon_root": "/unused", "sen2neon_rows": [{"id": "a"}],
                "selection_ids": ["a"], "raw_rows": []}
    options = {"every_n_train_steps": 1, "sen2neon": {"final_assessment": False}}

    def callback():
        cb = TransferValidation(prepared, options, tmp_path)
        def evaluate(self, trainer, module, rows, label, reuse=None):
            step = int(trainer.global_step)
            # Step 1 and 3 improve RGB but fail NIR; step 2 qualifies exactly at limits.
            lp = {0: .5, 1: .3, 2: .45, 3: .2, 4: .44}[step]
            nir = .12 if step in (1, 3) else (.1 if step == 0 else .11)
            return [{"id": "a", "site": "X", "landcover": "Forest",
                     "metrics": {"rgb_lpips": lp, "nir_mae": nir, "sam": .2 if step == 0 else .22}}]
        cb._evaluate = types.MethodType(evaluate, cb)
        return cb

    def module():
        return DiffusionTrainingModule(tiny_model_config, validation_num_samples=0, scheduler_patience=None,
                                       sharpness_config={"p2_gamma": 0, "latent_detail_weight": 0})

    def fit(cb, model, steps, resume=None):
        trainer = pl.Trainer(accelerator="cpu", devices=1, logger=False, callbacks=[cb], max_steps=steps,
                             max_epochs=-1, num_sanity_val_steps=0, limit_val_batches=1,
                             val_check_interval=1, check_val_every_n_epoch=None,
                             enable_model_summary=False, enable_progress_bar=False, default_root_dir=tmp_path)
        trainer.fit(model, datamodule=OpenSRDataModule(bundle.root, train_patch_size=16, validation_patch_size=16,
                                                     batch_size=1, num_workers=0, pin_memory=False),
                    ckpt_path=resume)
        return trainer

    cb, model = callback(), module()
    fit(cb, model, 3)
    assert cb.best_step == 2
    assert cb.baseline["nir_mae"] == .1
    native = load_checkpoint_payload(tmp_path / "checkpoints/native/best-inference.ckpt")
    resume = load_checkpoint_payload(tmp_path / "checkpoints/resume/best.ckpt")
    key = next(k for k in native["state_dict"] if k.startswith("model.diffusion_model."))
    shadow = "diffusion.model_ema." + key.removeprefix("model.").replace(".", "")
    assert torch.equal(native["state_dict"][key], resume["state_dict"][shadow])
    assert load_diffusion_weights(module().diffusion, tmp_path / "checkpoints/native/best-inference.ckpt").is_exact
    restored = callback()
    fit(restored, module(), 4, str(tmp_path / "checkpoints/resume/last.ckpt"))
    assert restored.best_step == 4
    assert restored.baseline == cb.baseline
    changed = callback()
    changed.identity = "changed"
    with pytest.raises(ValueError, match="changed validation"):
        changed.load_state_dict(cb.state_dict())
