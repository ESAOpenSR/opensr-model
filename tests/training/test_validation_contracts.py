"""Numerical scores, missing-data reporting, and persisted validation previews."""

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

from opensr_model.training.modules import metrics, optim
from opensr_model.training.validation import images
from opensr_model.training.validation import metrics as validation


@pytest.mark.parametrize("missing", [False, True])
def test_reconstruction_scores_have_known_error_and_honest_invalid_support(missing):
    target = torch.full((1, 4, 16, 16), 0.4)
    predicted = target + 0.1
    lr = F.avg_pool2d(target, 4)
    mask = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    if missing:
        mask[..., 8, 8] = False
        predicted[..., 8, 8] = 100
    perceptual_inputs = []

    def perceptual(a, b):
        perceptual_inputs.append((a, b))
        return (a - b).square().mean(1, keepdim=True)

    scores = validation.reconstruction_scores(predicted, target, lr, mask, perceptual)
    for band in ("rgb", "nir", "rgbn"):
        assert scores[f"{band}_mae"] == pytest.approx(0.1)
        assert scores[f"{band}_rmse"] == pytest.approx(0.1)
        assert scores[f"{band}_bias"] == pytest.approx(0.1)
        assert scores[f"{band}_psnr"] == pytest.approx(20)
        assert math.isfinite(scores[f"{band}_ssim"])
    assert scores["ndvi_mae"] == 0
    if missing:
        assert math.isnan(scores["rgb_lpips"])
        assert scores["lpips_valid_fraction"] == 0
        a, b = perceptual_inputs[0]
        torch.testing.assert_close(a[..., 8, 8], b[..., 8, 8])
    else:
        assert scores["rgb_lpips"] == pytest.approx(0.04)
        assert scores["lpips_valid_fraction"] == 1
        assert scores["rgbn_lr_consistency"] == pytest.approx(0.1)


def test_dark_spectra_report_undefined_ndvi_and_json_null():
    x = torch.full((1, 4, 16, 16), 0.001)
    scores = validation.reconstruction_scores(
        x,
        x,
        F.avg_pool2d(x, 4),
        torch.ones(1, 1, 16, 16, dtype=torch.bool),
        lambda a, b: (a - b).square().mean(1, keepdim=True),
    )
    assert math.isnan(scores["ndvi_mae"])
    safe = validation.finite_json({**scores, "infinite": float("inf"), "missing": None})
    assert safe["ndvi_mae"] is None and safe["infinite"] is None
    json.dumps(safe, allow_nan=False)
    assert scores["rgb_mae"] == 0 and scores["rgb_ssim"] == pytest.approx(1)


def test_paired_report_matches_ids_and_excludes_nonfinite_metrics():
    baseline = [
        {"id": "a", "site": "one", "metrics": {"error": 3.0, "absent": None}},
        {"id": "b", "site": "two", "metrics": {"error": 5.0, "absent": float("nan")}},
    ]
    candidate = [
        {"id": "b", "site": "two", "metrics": {"error": 3.0, "absent": 1.0}},
        {"id": "a", "site": "one", "metrics": {"error": 1.0, "absent": 2.0}},
        {"id": "new", "site": "three", "metrics": {"error": 100.0}},
    ]
    result = validation.paired_report(baseline, candidate, resamples=50, seed=4)
    assert result["paired_tiles"] == 2 and result["sites"] == 2
    assert result["metrics"] == {
        "error": {"mean_delta": -2.0, "site_balanced_delta": -2.0, "ci95": [-2.0, -2.0]}
    }
    assert validation.paired_report(baseline, []) == {
        "paired_tiles": 0,
        "sites": 0,
        "metrics": {},
    }
    assert validation.aggregate(candidate) == {
        "error": pytest.approx(104 / 3),
        "absent": 1.5,
    }


@pytest.mark.parametrize("tolerance", [-1, float("nan"), float("inf")])
def test_guardrail_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        validation.guardrail_decision({}, {}, tolerance)


def test_guardrail_missing_or_nonfinite_metrics_cannot_select_a_checkpoint():
    baseline = {"rgb_lpips": 0.3, "nir_mae": 0.1, "sam": 0.2}
    accepted, reasons = validation.guardrail_decision(
        {"rgb_lpips": float("nan"), "nir_mae": None}, baseline
    )
    assert not accepted
    assert set(reasons) == {"nonfinite_rgb_lpips", "nonfinite_nir_mae", "nonfinite_sam"}


@pytest.mark.parametrize("has_reference", [False, True])
def test_preview_files_share_reference_stretch_and_preserve_missing_hr(
    tmp_path, has_reference
):
    lr = np.linspace(0, 1, 4 * 8 * 8).reshape(4, 8, 8)
    hr = np.repeat(np.repeat(lr, 4, -1), 4, -2) if has_reference else None
    sr = np.full((4, 32, 32), 2.0)
    row = images.preview_row(
        tmp_path,
        {"id": "a/b:site", "site": "site", "zoom_center_hr": [0, 31]},
        lr,
        sr,
        hr,
    )
    assert len(row) == len(images.image_columns()) == 21
    assert row[:3] == ["a/b:site", "site", ""]
    assert row.count(None) == (0 if has_reference else 6)
    for column, path in zip(images.image_columns()[3:], row[3:]):
        if path is None:
            assert column.endswith("/HR")
            continue
        assert Path(path).parent == tmp_path
        with Image.open(path) as preview:
            assert preview.size == (32, 32)
            if column.endswith("/SR"):
                assert np.asarray(preview).min() == 255


@pytest.mark.parametrize("center", [[-1, 0], [32, 0], [0], [0, 0, 0]])
def test_preview_rejects_invalid_zoom_center(tmp_path, center):
    with pytest.raises(ValueError, match="zoom_center_hr"):
        images.preview_row(
            tmp_path,
            {"id": "bad", "zoom_center_hr": center},
            np.ones((4, 8, 8)),
            np.ones((4, 32, 32)),
        )


def test_no_remote_image_logging_for_unconfigured_or_non_primary_trainer(tmp_path):
    for rank in (True, False):
        trainer = SimpleNamespace(is_global_zero=rank, logger=None)
        assert images.wandb_run(trainer) is None
        assert (
            images.log_outputs(trainer, tmp_path, [], label="validation", step=2)
            is None
        )


def test_wandb_image_logging_packages_files_without_network(tmp_path, monkeypatch):
    run, artifact = Mock(), Mock()
    run.id = "run-id"
    api = SimpleNamespace(
        Artifact=Mock(return_value=artifact),
        Image=Mock(side_effect=lambda p: ("image", p)),
        Table=Mock(return_value="table"),
    )
    monkeypatch.setitem(__import__("sys").modules, "wandb", api)
    monkeypatch.setattr(images, "wandb_run", lambda trainer: run)
    (tmp_path / "sample.npz").write_bytes(b"array")
    (tmp_path / "metrics.json").write_text("{}")
    images.log_outputs(
        None, tmp_path, [["id", "site", "land", "image.png", None]], label="a/b", step=7
    )
    assert artifact.add_file.call_count == 2
    run.log_artifact.assert_called_once_with(artifact)
    assert run.log.call_args.args[0]["optimizer_step"] == 7
    assert api.Table.call_args.kwargs["data"][0][-1] is None


def test_masks_are_expanded_resized_and_channel_validated():
    reference = torch.ones(1, 4, 4, 4)
    torch.testing.assert_close(metrics._expanded_mask(None, reference), reference)
    mask = torch.tensor([[[True, False], [False, True]]])
    expanded = metrics._expanded_mask(mask, reference)
    assert expanded.shape == reference.shape
    assert expanded.sum() == 32
    assert metrics.select_channel_mask(mask, slice(0, 3), total_channels=4) is mask
    with pytest.raises(ValueError, match="shape"):
        metrics.select_channel_mask(torch.ones(2), slice(0, 3), total_channels=4)
    with pytest.raises(ValueError, match="same"):
        metrics.select_channel_mask(
            torch.ones(1, 2, 4, 4), slice(0, 3), total_channels=4
        )
    with pytest.raises(ValueError, match="mask"):
        metrics.spectral_angle(reference, reference, torch.ones(4, 4))
    with pytest.raises(ValueError, match="value_max"):
        metrics.compute_reconstruction_metrics(
            reference, reference, value_min=1, value_max=0
        )
    with pytest.raises(ValueError, match="same"):
        metrics.compute_rgb_nir_reconstruction_metrics(reference, reference[:, :3])


@pytest.mark.parametrize("value", [[], [float("nan")], [-0.01], [1.01]])
def test_invalid_clipping_telemetry_is_rejected(value):
    with pytest.raises(ValueError, match="clipped_fraction"):
        metrics.batch_clipped_fraction({"clipped_fraction": value}, torch.ones(1))


def test_plateau_scheduler_reduces_lr_to_configured_floor():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = optim.build_adamw([parameter], {"lr": 0.1})
    config = optim.build_scheduler(
        optimizer,
        {"name": "plateau", "patience": 0, "factor": 0.5, "min_lr": 0.02},
        total_steps=5,
        monitor="val/psnr",
    )
    assert config["monitor"] == "val/psnr" and config["interval"] == "epoch"
    for value in [1, 2, 2, 2, 2]:
        config["scheduler"].step(value)
    assert optimizer.param_groups[0]["lr"] == 0.02
