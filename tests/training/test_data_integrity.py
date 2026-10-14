"""Exercise the file and metadata checks that precede optimizer updates."""

import json

import numpy as np
import pytest
import rasterio
import tacoreader.v1
import torch
from rasterio.transform import from_origin

from opensr_model.training.data import taco
from opensr_model.training.validation import data


def write_raster(path, values, nodata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        count=len(values),
        height=values.shape[1],
        width=values.shape[2],
        dtype=values.dtype,
        nodata=nodata,
        crs="EPSG:32632",
        transform=from_origin(100, 200, 10, 10),
    ) as dst:
        dst.write(values)


def test_raw_manifest_resolves_paths_bands_and_reflectance(tmp_path):
    pixels = np.stack(
        [
            np.full((8, 8), value, dtype=np.uint16)
            for value in (1000, 2000, 3000, 15000, 5000)
        ]
    )
    write_raster(tmp_path / "scene.tif", pixels)
    manifest = tmp_path / "raw.json"
    manifest.write_text(
        json.dumps(
            [{"id": "scene", "path": "scene.tif", "bands": [3, 2, 1, 4], "scale": 1e-4}]
        )
    )
    rows = data.load_raw_manifest(manifest)
    assert rows[0]["path"] == str(tmp_path / "scene.tif")
    actual = data.read_raw(rows[0])
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual[:, 0, 0], torch.tensor([0.3, 0.2, 0.1, 1.0]))
    assert json.loads(manifest.read_text())[0]["path"] == "scene.tif"
    assert data.load_raw_manifest(None) == []


@pytest.mark.parametrize(
    "defect",
    [
        "not_list",
        "duplicate",
        "missing_bands",
        "duplicate_bands",
        "zero_scale",
        "nan_scale",
        "negative_scale",
    ],
)
def test_raw_manifest_rejects_invalid_contract_before_opening_raster(tmp_path, defect):
    row = {
        "id": "scene",
        "path": "not-opened.tif",
        "bands": [1, 2, 3, 4],
        "scale": 1e-4,
    }
    rows = [row]
    if defect == "not_list":
        rows = {}
    elif defect == "duplicate":
        rows *= 2
    elif defect == "missing_bands":
        row.pop("bands")
    elif defect == "duplicate_bands":
        row["bands"] = [1, 1, 3, 4]
    else:
        row["scale"] = {
            "zero_scale": 0,
            "nan_scale": float("nan"),
            "negative_scale": -1,
        }[defect]
    manifest = tmp_path / "raw.json"
    manifest.write_text(json.dumps(rows))
    with pytest.raises(ValueError):
        data.load_raw_manifest(manifest)


@pytest.mark.parametrize("invalid", [0, -1, 65535, float("nan"), float("inf"), -9999])
def test_raw_validation_checks_every_pixel_before_preview_cropping(tmp_path, invalid):
    pixels = np.full((4, 8, 8), 1000.0, dtype=np.float32)
    pixels[3, -1, -1] = invalid
    path = tmp_path / "scene.tif"
    write_raster(path, pixels, nodata=-9999)
    with pytest.raises(taco.InvalidPairError, match="full raster"):
        data.read_raw({"path": path, "bands": [1, 2, 3, 4], "scale": 1e-4})


@pytest.fixture
def neon_manifest(tmp_path):
    root = tmp_path / "neon"
    root.mkdir()
    row = {
        "id": "scene",
        "lr": "s2_l2a_10m/lr.tif",
        "hr": "neon_2.5m_linearized/hr.tif",
        "lr_sha256": "lr-hash",
        "hr_sha256": "hr-hash",
    }
    for name in ("lr", "hr"):
        path = root / row[name]
        path.parent.mkdir()
        path.touch()
    source = {
        **row,
        "bands": [
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B11",
            "B12",
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([row]))
    (root / "metadata.jsonl").write_text(json.dumps(source) + "\n")
    return root, manifest, row, source


def test_neon_resolver_uses_environment_and_verifies_catalog(
    neon_manifest, monkeypatch
):
    root, manifest, row, _ = neon_manifest
    monkeypatch.setenv("SEN2NEON_ROOT", str(root))
    assert data.resolve_sen2neon({"enabled": True, "manifest": str(manifest)}) == (
        root,
        [row],
        manifest,
    )


@pytest.mark.parametrize(
    "defect",
    [
        "empty",
        "duplicate",
        "bands",
        "path",
        "missing_lr",
        "missing_hr",
        "lr_checksum",
        "hr_checksum",
    ],
)
def test_neon_resolver_rejects_incomplete_or_mismatched_datasets(neon_manifest, defect):
    root, manifest, row, source = neon_manifest
    rows = [row]
    if defect == "empty":
        rows = []
    elif defect == "duplicate":
        rows *= 2
    elif defect == "bands":
        source["bands"].reverse()
    elif defect == "path":
        source["lr"] = "wrong.tif"
    elif defect.startswith("missing_"):
        (root / row[defect.removeprefix("missing_")]).unlink()
    else:
        row.pop(defect.replace("checksum", "sha256"))
    manifest.write_text(json.dumps(rows))
    (root / "metadata.jsonl").write_text(json.dumps(source) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        data.resolve_sen2neon({"enabled": True, "root": root, "manifest": manifest})


def test_neon_optional_and_required_roots(tmp_path):
    root = tmp_path / "missing"
    assert data.resolve_sen2neon({"enabled": "auto", "root": root}) == (None, [], None)
    assert data.resolve_sen2neon({"enabled": False, "root": root}) == (None, [], None)
    with pytest.raises(FileNotFoundError, match="Required"):
        data.resolve_sen2neon({"enabled": True, "root": root})
    with pytest.raises(ValueError, match="enabled"):
        data.resolve_sen2neon({"enabled": "yes"})


@pytest.mark.parametrize("count", [0, -1, 11])
def test_diverse_selection_rejects_impossible_counts(count):
    with pytest.raises(ValueError, match="subset_size"):
        data.select_diverse([{}] * 10, count)


def test_diverse_selection_is_order_independent_and_covers_sites():
    rows = [
        {
            "id": f"{site}-{index}",
            "site": site,
            "landcover": "Developed" if (site, index) == ("a", 0) else "Forest",
        }
        for site in "abc"
        for index in range(4)
    ]
    chosen = data.select_diverse(rows, 8)
    assert chosen == data.select_diverse(list(reversed(rows)), 8)
    assert len({row["id"] for row in chosen}) == 8
    assert {row["site"] for row in chosen} == {"a", "b", "c"}
    assert chosen[0]["landcover"] == "Developed"


@pytest.mark.parametrize(
    "payload,match",
    [
        (b"short", "Incomplete TACO header"),
        (b"XX" + bytes(16), "Invalid TACO v1 magic"),
        (b"#y" + bytes(16), "Incomplete TACO footer"),
    ],
)
def test_invalid_taco_headers_are_rejected(tmp_path, payload, match):
    path = tmp_path / "bad.tortilla"
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        taco.resolve_taco_paths(path)


@pytest.mark.parametrize(
    "value,error", [([], ValueError), (42, TypeError), ([None], TypeError)]
)
def test_taco_path_argument_validation(value, error):
    with pytest.raises(error):
        taco.resolve_taco_paths(value)


def test_taco_discovery_refuses_hidden_and_empty_sources(tmp_path):
    with pytest.raises(FileNotFoundError, match="No complete"):
        taco.resolve_taco_paths(tmp_path)
    hidden = tmp_path / ".upload.taco"
    hidden.touch()
    with pytest.raises(ValueError, match="Hidden upload"):
        taco.resolve_taco_paths(hidden)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        taco.resolve_taco_paths(tmp_path / "absent.taco")


@pytest.mark.parametrize(
    "defect,match",
    [
        ("columns", "missing columns"),
        ("empty", "no samples"),
        ("blank_id", "Empty outer"),
        ("format", "must be TORTILLA"),
        ("duplicate", "Duplicate outer"),
    ],
)
def test_taco_catalog_rejects_bad_sample_metadata(fake_taco_factory, defect, match):
    bundle = fake_taco_factory()
    table = tacoreader.v1.load(str(bundle.taco_paths[0]))
    if defect == "columns":
        table.columns = ["tortilla:id"]
    elif defect == "empty":
        table._rows.clear()
    elif defect == "blank_id":
        table._rows[0]["tortilla:id"] = "  "
    elif defect == "format":
        table._rows[0]["tortilla:file_format"] = "GTiff"
    else:
        table._rows[1]["tortilla:id"] = table._rows[0]["tortilla:id"]
    with pytest.raises(ValueError, match=match):
        taco.scan_taco_catalog(bundle.taco_paths)


@pytest.mark.parametrize(
    "defect,match",
    [
        ("removed", "changed after catalog snapshot"),
        ("renamed", "changed after snapshot"),
        ("read_error", "Failed to read nested"),
    ],
)
def test_dataset_detects_archive_change_after_snapshot(
    fake_taco_factory, monkeypatch, defect, match
):
    bundle = fake_taco_factory()
    dataset = taco.TacoPairedDataset(bundle.root, train_patch_size=16)
    assert dataset.num_samples == 12
    table = tacoreader.v1.load(str(bundle.taco_paths[0]))
    if defect == "removed":
        table._rows.clear()
    elif defect == "renamed":
        table._rows[0]["tortilla:id"] = "new-id"
    else:

        def fail(*args):
            raise OSError("damaged archive")

        monkeypatch.setattr(table, "read", fail)
    with pytest.raises(RuntimeError, match=match):
        dataset[0]


@pytest.mark.parametrize(
    "sample,expected",
    [
        ("field_custom", "tile:field"),
        ("field_google", "tile:field"),
        ("field_20260301", "tile:field"),
        ("standalone", "sample:standalone"),
    ],
)
def test_grouping_prevents_provider_and_temporal_split_leakage(sample, expected):
    assert taco._group_key({"hr_product": "custom"}, sample) == expected


@pytest.mark.parametrize(
    "defect,match",
    [
        ("bands", "must contain 4 bands"),
        ("dtype", "must be uint16"),
        ("nodata", "nodata must be"),
        ("crs", "no CRS"),
        ("different_crs", "do not share a CRS"),
        ("shape", "not exactly 4x"),
        ("corrupt", "Cannot read"),
    ],
)
def test_dataset_rejects_invalid_raster_contracts(fake_taco_factory, defect, match):
    bundle = fake_taco_factory(schema="crosssensor", include_nodata=False)
    dataset = taco.TacoPairedDataset(bundle.root, training=False)
    path = bundle.raster_paths["hr"]
    with rasterio.open(path) as source:
        profile, values = source.profile.copy(), source.read()
    if defect == "corrupt":
        path.write_bytes(b"corrupt raster")
    else:
        if defect == "bands":
            values = values[:3]
            profile["count"] = 3
        elif defect == "dtype":
            values = values.astype(np.float32)
            profile["dtype"] = "float32"
        elif defect == "nodata":
            profile["nodata"] = None
        elif defect == "crs":
            profile["crs"] = None
        elif defect == "different_crs":
            profile["crs"] = "EPSG:4326"
        else:
            values = values[:, :-4]
            profile["height"] = values.shape[1]
        path.unlink()
        with rasterio.open(path, "w", **profile) as destination:
            destination.write(values)
    with pytest.raises((ValueError, TypeError, RuntimeError), match=match):
        dataset[0]


@pytest.mark.parametrize(
    "defect,match",
    [
        ("columns", "missing child columns"),
        ("blank", "empty child id"),
        ("duplicate", "duplicate asset id"),
        ("format", "must be GTiff"),
        ("path", "did not resolve to a path"),
        ("unknown", "unsupported asset ids"),
    ],
)
def test_dataset_rejects_malformed_child_assets(fake_taco_factory, defect, match):
    bundle = fake_taco_factory(schema="crosssensor")
    table = tacoreader.v1.load(str(bundle.taco_paths[0])).read(0)
    if defect == "columns":
        table.columns = ["tortilla:id"]
    elif defect == "blank":
        table._rows[0]["tortilla:id"] = None
    elif defect == "duplicate":
        table._rows[1]["tortilla:id"] = table._rows[0]["tortilla:id"]
    elif defect == "format":
        table._rows[0]["tortilla:file_format"] = "PNG"
    elif defect == "path":
        table._reads[0] = None
    else:
        table._rows[0]["tortilla:id"] = "unknown"
    with pytest.raises((ValueError, TypeError), match=match):
        taco.TacoPairedDataset(bundle.root, training=False)[0]


@pytest.mark.parametrize(
    "defect,match",
    [
        ("missing", "Missing synthetic HR NIR"),
        ("corrupt", "Cannot read synthetic"),
        ("keys", "exactly key"),
        ("shape", "must have shape"),
        ("dtype", "must be float16"),
        ("nan", "non-finite"),
        ("negative", "already be in"),
        ("too_large", "already be in"),
    ],
)
def test_synthetic_nir_data_must_match_training_contract(
    fake_taco_factory, defect, match
):
    bundle = fake_taco_factory()
    dataset = taco.TacoPairedDataset(bundle.root, training=False)
    path = bundle.synthetic_hr_nir_dir / f"{dataset.records[0].sample_id}.npz"
    array = np.full((1, 32, 32), 0.4, dtype=np.float16)
    if defect == "missing":
        path.unlink()
    elif defect == "corrupt":
        path.write_bytes(b"corrupt npz")
    else:
        if defect == "shape":
            array = array[0]
        elif defect == "dtype":
            array = array.astype(np.float32)
        elif defect in ("nan", "negative", "too_large"):
            array[0, 0, 0] = {"nan": float("nan"), "negative": -0.1, "too_large": 1.1}[
                defect
            ]
        np.savez_compressed(path, **{"wrong" if defect == "keys" else "nir": array})
    with pytest.raises((ValueError, TypeError, FileNotFoundError), match=match):
        dataset[0]


@pytest.mark.parametrize(
    "defect,match",
    [
        ("missing", "manifest is missing"),
        ("corrupt", "manifest is unreadable"),
        ("output", "output_dir does not match"),
        ("normalization", "no normalization contract"),
        ("dtype", "float16 outputs"),
        ("archives", "no archive snapshot"),
        ("malformed_archive", "snapshot is malformed"),
        ("archive_size", "different TACO archive"),
        ("catalog_count", "sample count does not match"),
        ("completed_count", "every TACO sample"),
    ],
)
def test_synthetic_nir_manifest_must_match_exact_archive_snapshot(
    fake_taco_factory, defect, match
):
    bundle = fake_taco_factory()
    path = bundle.synthetic_hr_nir_dir / "inference_manifest.json"
    manifest = json.loads(path.read_text())
    if defect == "missing":
        path.unlink()
    elif defect == "corrupt":
        path.write_text("{incomplete")
    else:
        if defect == "output":
            manifest["output_dir"] = str(bundle.root)
        elif defect == "normalization":
            manifest.pop("normalization")
        elif defect == "dtype":
            manifest["normalization"]["output_dtype"] = "float32"
        elif defect == "archives":
            manifest.pop("archives")
        elif defect == "malformed_archive":
            manifest["archives"] = [{}]
        elif defect == "archive_size":
            manifest["archives"][0]["size"] += 1
        elif defect == "catalog_count":
            manifest["catalog_samples"] += 1
        else:
            manifest["completed"] -= 1
        path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        taco.OpenSRDataModule(bundle.root).setup("fit")


def test_crop_fallback_finds_valid_support_and_keeps_lr_hr_alignment():
    lr = torch.arange(64.0).reshape(1, 8, 8)
    hr = lr.repeat_interleave(4, -2).repeat_interleave(4, -1)
    mask = torch.zeros_like(hr, dtype=torch.bool)
    mask[0, 0, 0] = True
    cropped_hr, cropped_lr, cropped_mask = taco._aligned_crop(
        hr, lr, mask, patch=(8, 8), factor=4, random=False
    )
    assert cropped_mask.any()
    torch.testing.assert_close(
        cropped_hr, cropped_lr.repeat_interleave(4, -2).repeat_interleave(4, -1)
    )
    with pytest.raises(ValueError, match="no valid pixels"):
        taco._aligned_crop(
            hr, lr, torch.zeros_like(mask), patch=(8, 8), factor=4, random=False
        )


def test_clipping_counts_only_valid_support_and_can_be_disabled():
    values = torch.tensor([[[-1.0, 0.5], [2.0, 99.0]]])
    valid = torch.tensor([[[True, True], [True, False]]])
    bounded, clipped, count = taco._clamp_and_count(
        values, valid, (0.0, 1.0), sample_id="scene", name="HR"
    )
    torch.testing.assert_close(bounded, torch.tensor([[[0.0, 0.5], [1.0, 99.0]]]))
    assert (clipped, count) == (2, 3)
    unchanged, clipped, count = taco._clamp_and_count(
        values, valid, None, sample_id="scene", name="HR"
    )
    assert unchanged is values and (clipped, count) == (0, 3)
    with pytest.raises(ValueError, match="no valid pixels"):
        taco._clamp_and_count(
            values, torch.zeros_like(valid), None, sample_id="scene", name="HR"
        )
    with pytest.raises(ValueError, match="non-finite"):
        taco._clamp_and_count(
            values * float("nan"), valid, None, sample_id="scene", name="HR"
        )
