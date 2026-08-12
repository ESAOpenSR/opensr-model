from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from rasterio.transform import from_origin

from opensr_model.training.data import (
    OpenSRDataModule,
    TacoPairedDataset,
    resolve_taco_paths,
    scan_taco_catalog,
    split_taco_records,
)


def test_directory_discovers_only_visible_complete_tortillas(fake_taco_factory) -> None:
    bundle = fake_taco_factory(parts=2)
    (bundle.root / ".upload.part.tortilla").write_bytes(b"in progress")
    incomplete = bundle.root / "unfinished.part.tortilla"
    incomplete.write_bytes(
        b"#y" + (100).to_bytes(8, "little") + (20).to_bytes(8, "little")
    )
    csv_path = bundle.root / "manifest.csv"
    csv_path.write_text("hr,lr\n")

    with pytest.warns(RuntimeWarning, match="not-yet-complete"):
        paths = resolve_taco_paths(bundle.root)

    assert paths == tuple(sorted(bundle.taco_paths))
    with pytest.raises(ValueError, match="Incomplete TACO footer"):
        resolve_taco_paths(incomplete)
    with pytest.raises(ValueError, match="Only TACO"):
        resolve_taco_paths(csv_path)


def test_assets_use_synthetic_nir_for_both_lr_and_hr(fake_taco_factory) -> None:
    bundle = fake_taco_factory(
        include_nodata=False,
        child_order=("hr", "lrharm", "hrharm", "lr"),
    )
    dataset = TacoPairedDataset(bundle.root, training=False)

    sample = dataset[0]

    assert set(sample) == {
        "image",
        "LR_image",
        "valid_mask",
        "sample_id",
        "clipped_fraction",
    }
    assert sample["image"].shape == (4, 32, 32)
    assert sample["LR_image"].shape == (4, 8, 8)
    assert sample["valid_mask"].shape == (1, 32, 32)
    assert torch.allclose(
        sample["LR_image"][:, 2, 2],
        torch.tensor([0.1, 0.2, 0.3, 0.45]),
        atol=1e-3,
    )
    assert torch.allclose(
        sample["image"][:, 8, 8],
        torch.tensor([0.11, 0.21, 0.31, 0.45]),
        atol=1e-3,
    )
    assert sample["LR_image"][3, -1, -1] == pytest.approx(0.45, abs=1e-3)
    assert sample["valid_mask"][0, -1, -1] == 1
    assert sample["clipped_fraction"] == 0


def test_sen2naipv2_crosssensor_taco_reads_direct_rgb_nir_pair(
    fake_taco_factory,
) -> None:
    bundle = fake_taco_factory(schema="crosssensor", include_nodata=False)

    assert resolve_taco_paths(bundle.root) == bundle.taco_paths
    sample = TacoPairedDataset(bundle.root, training=False)[0]

    assert sample["image"].shape == (4, 32, 32)
    assert sample["LR_image"].shape == (4, 8, 8)
    assert sample["valid_mask"].shape == (1, 32, 32)
    assert torch.allclose(
        sample["LR_image"][:, 2, 2], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    assert torch.allclose(
        sample["image"][:, 8, 8], torch.tensor([0.11, 0.21, 0.31, 0.41])
    )
    assert sample["clipped_fraction"] == 0


def test_nodata_is_zero_filled_with_conservative_pair_mask(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=True)
    sample = TacoPairedDataset(bundle.root, training=False)[0]

    assert torch.all(sample["LR_image"][:, 0, 0] == 0)
    assert sample["valid_mask"][0, 0, 0] == 0
    assert torch.all(sample["image"][:, 0, 0] == 0)
    # Direct HR RGB nodata is combined with conservative LR support.
    assert sample["valid_mask"][0, 4, 4] == 0
    assert torch.all(sample["image"][:, 4, 4] == 0)
    assert sample["LR_image"][3, -1, -1] == pytest.approx(0.45, abs=1e-3)
    assert sample["valid_mask"][0, -1, -1] == 1


def test_valid_outliers_are_clamped_and_reported(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=False, include_outliers=True)

    sample = TacoPairedDataset(bundle.root, training=False)[0]

    assert sample["image"].max() <= 1
    assert sample["LR_image"].max() <= 1
    assert 0 < sample["clipped_fraction"] < 1


def test_provider_variants_never_cross_stable_split(fake_taco_factory) -> None:
    sample_ids = [
        "tile_A_bing",
        "tile_A_google",
        "tile_B_bing",
        "tile_B_esri",
        "tile_C_google",
        "tile_C_esri",
    ]
    # Deliberately different acquisitions: grouping must use the spatial base,
    # not id_gee, to keep provider variants together.
    bundle = fake_taco_factory(
        sample_ids=sample_ids,
        id_gees=[f"acquisition-{index}" for index in range(len(sample_ids))],
        include_nodata=False,
    )
    records = scan_taco_catalog(resolve_taco_paths(bundle.root))

    train, val = split_taco_records(records, val_fraction=0.4, seed=7)
    train_ids = {record.sample_id for record in train}
    val_ids = {record.sample_id for record in val}
    for base in ("tile_A", "tile_B", "tile_C"):
        variants = {sample_id for sample_id in sample_ids if sample_id.startswith(base)}
        assert variants <= train_ids or variants <= val_ids
    train_again, val_again = split_taco_records(records, val_fraction=0.4, seed=7)
    assert train == train_again
    assert val == val_again


def test_sen2naipv2_dates_for_one_location_never_cross_split(
    fake_taco_factory,
) -> None:
    sample_ids = [
        "NA5120_E1000N1000__m_patch_20200101",
        "NA5120_E1000N1000__m_patch_20220101",
        "NA5120_E1001N1001__m_patch_20200101",
        "NA5120_E1001N1001__m_patch_20220101",
        "NA5120_E1002N1002__m_patch_20200101",
        "NA5120_E1002N1002__m_patch_20220101",
    ]
    bundle = fake_taco_factory(
        schema="crosssensor", sample_ids=sample_ids, include_nodata=False
    )
    records = scan_taco_catalog(resolve_taco_paths(bundle.root))

    train, val = split_taco_records(records, val_fraction=0.4, seed=7)
    train_ids = {record.sample_id for record in train}
    val_ids = {record.sample_id for record in val}
    for location in ("E1000N1000", "E1001N1001", "E1002N1002"):
        views = {sample_id for sample_id in sample_ids if location in sample_id}
        assert views <= train_ids or views <= val_ids


def test_datamodule_crops_training_but_validates_complete_images(
    fake_taco_factory,
) -> None:
    bundle = fake_taco_factory(include_nodata=False, parts=2)
    module = OpenSRDataModule(
        bundle.root,
        val_fraction=0.35,
        split_seed=11,
        train_patch_size=16,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotate_90=False,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    module.setup("fit")
    train_batch = next(iter(module.train_dataloader()))
    val_batch = next(iter(module.val_dataloader()))

    assert module.taco_paths == tuple(sorted(bundle.taco_paths))
    assert module.num_samples == 12
    assert module.train_sample_count + module.val_sample_count == 12
    assert train_batch["image"].shape == (2, 4, 16, 16)
    assert train_batch["LR_image"].shape == (2, 4, 4, 4)
    assert val_batch["image"].shape[-2:] == (32, 32)
    assert val_batch["LR_image"].shape[-2:] == (8, 8)
    assert val_batch["clipped_fraction"].shape[0] == len(val_batch["sample_id"])


def test_datamodule_can_validate_deterministic_aligned_crops(
    fake_taco_factory,
) -> None:
    bundle = fake_taco_factory(include_nodata=False, parts=2)
    module = OpenSRDataModule(
        bundle.root,
        val_fraction=0.35,
        split_seed=11,
        train_patch_size=16,
        validation_patch_size=16,
        horizontal_flip_probability=0.0,
        vertical_flip_probability=0.0,
        rotate_90=False,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    module.setup("fit")
    first = module.val_dataset[0]
    second = module.val_dataset[0]

    assert first["image"].shape == (4, 16, 16)
    assert first["LR_image"].shape == (4, 4, 4)
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["LR_image"], second["LR_image"])


def test_misaligned_transform_is_rejected(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    import rasterio

    with rasterio.open(bundle.raster_paths["hrharm"], "r+") as dataset:
        dataset.transform = from_origin(100.0, 200.0, 3.0, 3.0)

    dataset = TacoPairedDataset(bundle.root, training=False)
    with pytest.raises(ValueError, match="exact 4x grid"):
        dataset[0]


def test_reader_is_cached_lazily_within_a_process(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    dataset = TacoPairedDataset(bundle.root, training=False)
    calls_after_catalog = len(bundle.load_calls)

    dataset[0]
    calls_after_first_read = len(bundle.load_calls)
    dataset[1]

    assert calls_after_first_read == calls_after_catalog + 1
    assert len(bundle.load_calls) == calls_after_first_read


def test_unsupported_hr_nir_strategy_is_rejected(fake_taco_factory) -> None:
    bundle = fake_taco_factory()
    with pytest.raises(ValueError, match="synthetic_sidecar"):
        TacoPairedDataset(bundle.root, hr_nir_strategy="invent_hr")


def test_incomplete_synthetic_nir_generation_is_rejected(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    assert bundle.synthetic_hr_nir_dir is not None
    manifest_path = bundle.synthetic_hr_nir_dir / "inference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    module = OpenSRDataModule(bundle.root, num_workers=0, pin_memory=False)
    with pytest.raises(RuntimeError, match="not complete"):
        module.setup("fit")


def test_missing_synthetic_nir_sidecar_is_rejected_before_fit(
    fake_taco_factory,
) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    assert bundle.synthetic_hr_nir_dir is not None
    (bundle.synthetic_hr_nir_dir / "tile_000_bing.npz").unlink()

    module = OpenSRDataModule(bundle.root, num_workers=0, pin_memory=False)
    with pytest.raises(FileNotFoundError, match="Missing synthetic HR NIR sidecars"):
        module.setup("fit")


def test_malformed_synthetic_nir_is_rejected_lazily(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    assert bundle.synthetic_hr_nir_dir is not None
    np.savez_compressed(
        bundle.synthetic_hr_nir_dir / "tile_000_bing.npz",
        nir=np.zeros((1, 16, 16), dtype=np.float32),
    )

    dataset = TacoPairedDataset(bundle.root, training=False)
    with pytest.raises(ValueError, match="must have shape"):
        dataset[0]


def test_synthetic_nir_is_downsampled_and_blurred_without_using_native_b8(
    fake_taco_factory,
) -> None:
    bundle = fake_taco_factory(include_nodata=False)
    assert bundle.synthetic_hr_nir_dir is not None
    lr_values = np.linspace(1000, 8000, 8 * 8, dtype=np.uint16).reshape(8, 8)
    import rasterio

    with rasterio.open(bundle.raster_paths["lr"], "r+") as dataset:
        dataset.write(lr_values, indexes=8)
    synthetic = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(1, 32, 32)
    synthetic = np.square(synthetic).astype(np.float16)
    np.savez_compressed(
        bundle.synthetic_hr_nir_dir / "tile_000_bing.npz", nir=synthetic
    )

    sample = TacoPairedDataset(bundle.root, training=False)[0]
    synthetic_tensor = torch.from_numpy(synthetic.astype(np.float32))
    resized = torch.nn.functional.interpolate(
        synthetic_tensor.unsqueeze(0),
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    )
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]]).div(16.0)
    expected_lr = torch.nn.functional.conv2d(
        torch.nn.functional.pad(resized, (1, 1, 1, 1), mode="replicate"),
        kernel.view(1, 1, 3, 3),
    ).squeeze(0)

    assert torch.equal(sample["image"][3], synthetic_tensor[0])
    assert torch.allclose(sample["LR_image"][3:4], expected_lr)
    native_b8 = torch.from_numpy(lr_values.astype(np.float32)).div(1e4)
    assert not torch.allclose(sample["LR_image"][3], native_b8)


def test_validation_loader_uses_one_fixed_randomized_record_order(
    fake_taco_factory,
) -> None:
    bundle = fake_taco_factory(include_nodata=False, parts=3)
    split_seed = 23
    module = OpenSRDataModule(
        bundle.root,
        val_fraction=0.5,
        split_seed=split_seed,
        val_shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    module.setup("validate")

    assert module.sample_records is not None
    assert module.val_dataset is not None
    _, original_validation = split_taco_records(
        module.sample_records, val_fraction=0.5, seed=split_seed
    )
    generator = torch.Generator().manual_seed(split_seed + 2_000_003)
    order = torch.randperm(len(original_validation), generator=generator).tolist()
    expected = [original_validation[index].sample_id for index in order]
    actual = [record.sample_id for record in module.val_dataset.records]

    assert actual == expected
    first_pass = [batch["sample_id"] for batch in module.val_dataloader()]
    second_pass = [batch["sample_id"] for batch in module.val_dataloader()]
    assert first_pass == second_pass
