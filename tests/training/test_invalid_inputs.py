"""Validate user-controlled data and training settings before expensive work."""

from types import SimpleNamespace

import pytest
import torch

from opensr_model.training.data import taco
from opensr_model.training.modules import diffusion


@pytest.mark.parametrize(
    "options,message",
    [
        ({"factor": 2}, "factor=4"),
        ({"lr_asset_id": " "}, "non-empty"),
        ({"hr_asset_id": "lrharm"}, "must differ"),
        ({"nir_band": 0}, "nir_band"),
        ({"nir_band": 13}, "nir_band"),
        ({"hr_nir_strategy": "interpolate"}, "synthetic_sidecar"),
        ({"reflectance_scale": float("nan")}, "reflectance_scale"),
        ({"reflectance_scale": 0}, "reflectance_scale"),
        ({"nodata": 0}, "nodata=65535"),
        ({"horizontal_flip_probability": -1}, "probability"),
        ({"vertical_flip_probability": 2}, "probability"),
    ],
)
def test_dataset_rejects_invalid_settings_before_opening_files(
    tmp_path, options, message
):
    with pytest.raises(ValueError, match=message):
        taco.TacoPairedDataset(taco_path=tmp_path / "nonexistent.taco", **options)


@pytest.mark.parametrize(
    "options,message",
    [
        ({"val_fraction": 0}, "val_fraction"),
        ({"val_fraction": 1}, "val_fraction"),
        ({"batch_size": 0}, "batch sizes"),
        ({"val_batch_size": -1}, "batch sizes"),
        ({"num_workers": -1}, "num_workers"),
        ({"persistent_workers": True}, "num_workers"),
        ({"prefetch_factor": 2}, "prefetch_factor"),
        ({"num_workers": 1, "prefetch_factor": 0}, "prefetch_factor"),
    ],
)
def test_datamodule_rejects_invalid_loader_options(tmp_path, options, message):
    with pytest.raises(ValueError, match=message):
        taco.OpenSRDataModule(taco_path=tmp_path / "nonexistent.taco", **options)


@pytest.mark.parametrize(
    "value,error",
    [([1, 2], ValueError), ([1, 2, 4], ValueError), ([1.0, 2, 3], ValueError)],
)
def test_rgb_band_selection_requires_one_based_permutation(value, error):
    with pytest.raises(error):
        taco._coerce_rgb_bands(value)
    assert taco._coerce_rgb_bands([3, 1, 2]) == (3, 1, 2)


@pytest.mark.parametrize(
    "value,error",
    [
        (0, ValueError),
        ((4, 6), ValueError),
        ((4, 4, 4), TypeError),
        (object(), TypeError),
    ],
)
def test_patch_validation(value, error):
    with pytest.raises(error):
        taco._coerce_patch(value)
    assert taco._coerce_patch([8, 12]) == (8, 12)
    assert taco._coerce_patch(None) is None


@pytest.mark.parametrize("value", [[1], [1, 0], [float("nan"), 1], [0, float("inf")]])
def test_range_validation(value):
    with pytest.raises(ValueError, match="value_range"):
        taco._coerce_range(value)
    assert taco._coerce_range(None) is None


def test_group_split_rejects_leakage_and_preserves_groups_at_extreme_fractions(
    tmp_path,
):
    records = [
        taco.TacoSampleRecord(tmp_path / "data.taco", i, str(i), f"group-{i//2}")
        for i in range(6)
    ]
    with pytest.raises(ValueError, match="independent"):
        taco.split_taco_records(records[:2], val_fraction=0.2, seed=1)
    with pytest.raises(ValueError, match="val_fraction"):
        taco.split_taco_records(records, val_fraction=1, seed=1)
    for fraction in (1e-20, 1 - 1e-12):
        train, val = taco.split_taco_records(records, val_fraction=fraction, seed=1)
        assert train and val and len(train) + len(val) == 6
        assert {r.split_group for r in train}.isdisjoint({r.split_group for r in val})


@pytest.mark.parametrize(
    "options,message",
    [
        ({"learning_rate": 0}, "learning_rate"),
        ({"weight_decay": -1}, "weight_decay"),
        ({"scheduler_patience": -1}, "scheduler_patience"),
        ({"scheduler_factor": 1}, "scheduler_factor"),
        ({"validation_num_samples": -1}, "counts"),
        ({"validation_image_batches": -1}, "counts"),
        ({"validation_sample_every_n_epochs": 0}, "every_n_epochs"),
        ({"validation_sampling_steps": 0}, "sampling_steps"),
        ({"validation_sampling_eta": -1}, "sampling_eta"),
        ({"validation_sampling_temperature": -1}, "temperature"),
        ({"validation_detail_crop_size": 7}, "divisible by four"),
    ],
)
def test_diffusion_training_rejects_invalid_hyperparameters(options, message):
    with pytest.raises(ValueError, match=message):
        diffusion.DiffusionTrainingModule({}, **options)


@pytest.mark.parametrize(
    "field,value",
    [
        ("unknown", 1),
        ("p2_gamma", -1),
        ("latent_detail_weight", float("nan")),
        ("p2_k", 0),
        ("latent_detail_max_timestep_fraction", 2),
        ("decoded_max_timestep_fraction", -1),
        ("decoded_every_n_batches", 0),
        ("decoded_max_batch_size", 0),
        ("perceptual_backbone", ""),
    ],
)
def test_sharpness_options_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        diffusion._resolve_sharpness_config({field: value})


def test_nested_config_supports_mapping_and_attribute_sources():
    assert diffusion._config_value({"a": {"b": 2}}, "a", "b") == 2
    assert (
        diffusion._config_value(SimpleNamespace(a=SimpleNamespace(b=3)), "a", "b") == 3
    )
    for value in ({}, SimpleNamespace()):
        assert diffusion._config_value(value, "missing", default=5) == 5
        with pytest.raises(KeyError, match="missing"):
            diffusion._config_value(value, "missing")


@pytest.mark.parametrize("shape", [(2, 1, 4, 4), (1, 1, 2, 2), (1, 2, 4, 4), (1, 4)])
def test_per_sample_mask_rejects_misaligned_data(shape):
    with pytest.raises(ValueError, match="mask"):
        diffusion._masked_per_sample_mean(torch.ones(1, 4, 4, 4), torch.ones(shape))


def test_per_sample_masks_keep_individual_denominators():
    values = torch.stack([torch.full((4, 4, 4), 2.0), torch.full((4, 4, 4), 5.0)])
    mask = torch.ones(2, 4, 4)
    mask[1] = 0
    torch.testing.assert_close(
        diffusion._masked_per_sample_mean(values, mask), torch.tensor([2.0, 0.0])
    )
    with pytest.raises(ValueError, match="BCHW"):
        diffusion._laplacian(torch.ones(4, 4))
