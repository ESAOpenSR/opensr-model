from __future__ import annotations

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


def test_assets_are_selected_by_id_and_normalized_to_rgb_nir(fake_taco_factory) -> None:
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
        sample["LR_image"][:, 2, 2], torch.tensor([0.1, 0.2, 0.3, 0.4])
    )
    assert torch.allclose(
        sample["image"][:, 8, 8], torch.tensor([0.11, 0.21, 0.31, 0.4])
    )
    assert sample["LR_image"][3, -1, -1] == 0
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


def test_nodata_is_zero_filled_with_conservative_hr_nir_mask(fake_taco_factory) -> None:
    bundle = fake_taco_factory(include_nodata=True)
    sample = TacoPairedDataset(bundle.root, training=False)[0]

    assert torch.all(sample["LR_image"][:, 0, 0] == 0)
    assert sample["valid_mask"][0, 0, 0] == 0
    assert torch.all(sample["image"][:, 0, 0] == 0)
    # Direct HR RGB nodata is combined with the upsampled LR validity.
    assert sample["valid_mask"][0, 4, 4] == 0
    assert torch.all(sample["image"][:, 4, 4] == 0)
    # A digital number of zero is data, not nodata.
    assert sample["LR_image"][3, -1, -1] == 0
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
    with pytest.raises(ValueError, match="upsample_lr"):
        TacoPairedDataset(bundle.root, hr_nir_strategy="invent_hr")
