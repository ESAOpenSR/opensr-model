from __future__ import annotations

import pytest
import torch

from opensr_model.utils import (
    apply_no_data_mask,
    assert_tensor_validity,
    create_no_data_mask,
    linear_transform_4b,
    linear_transform_6b,
    linear_transform_placeholder,
    plot_example,
    plot_uncertainty,
    revert_padding,
    suppress_stdout,
)


def test_four_band_normalization_round_trips_bchw_and_chw() -> None:
    image = torch.tensor([0.03, 0.06, 0.09, 0.2]).reshape(4, 1, 1).expand(4, 8, 8)
    normalized = linear_transform_4b(image.clone(), stage="norm")
    restored = linear_transform_4b(normalized.clone(), stage="denorm")

    assert normalized.shape == image.shape
    assert torch.allclose(restored, image, atol=1e-7)
    batch = image.unsqueeze(0)
    assert torch.allclose(
        linear_transform_4b(
            linear_transform_4b(batch.clone(), stage="norm"), stage="denorm"
        ),
        batch,
        atol=1e-7,
    )
    assert linear_transform_placeholder(image) is image


def test_four_band_normalization_handles_multi_image_tensor() -> None:
    image = torch.tensor([0.03, 0.06, 0.09, 0.2]).reshape(1, 4, 1, 1)
    multi_image = image.expand(2, 8, 4, 4, 4).clone()
    expected = multi_image.clone()

    normalized = linear_transform_4b(multi_image, stage="norm")
    restored = linear_transform_4b(normalized, stage="denorm")

    assert restored.shape == expected.shape
    assert torch.allclose(restored, expected, atol=1e-7)


def test_six_band_normalization_round_trips_channel_last_values() -> None:
    image = torch.full((2, 4, 4, 6), 0.25)
    restored = linear_transform_6b(
        linear_transform_6b(image.clone(), stage="norm"), stage="denorm"
    )
    assert torch.allclose(restored, image)


@pytest.mark.parametrize("height,width", [(100, 100), (100, 128), (127, 101)])
def test_tensor_validation_sanitizes_pads_and_padding_is_reversible(
    height, width, capsys
) -> None:
    image = torch.ones(4, height, width)
    image[0, 0, 0] = torch.nan
    image[1, 0, 0] = torch.inf
    prepared, padding = assert_tensor_validity(image)

    assert prepared.shape == (1, 4, 128, 128)
    assert torch.isfinite(prepared).all()
    left, right, top, bottom = padding
    assert left + right == 128 - width
    assert top + bottom == 128 - height
    super_resolved = torch.arange(4 * 512 * 512).reshape(1, 4, 512, 512)
    reverted = revert_padding(super_resolved, padding)
    assert reverted.shape == (1, 4, height * 4, width * 4)
    assert torch.equal(
        reverted,
        super_resolved[
            :, :, top * 4 : (128 - bottom) * 4, left * 4 : (128 - right) * 4
        ],
    )
    message = capsys.readouterr().out
    assert f"reflect-padding {height}x{width} LR input to 128x128" in message


@pytest.mark.parametrize(
    "shape", [(4, 128, 128), (2, 4, 128, 128), (128, 128, 4), (2, 128, 128, 4)]
)
def test_tensor_validation_accepts_exact_patches_without_message(shape, capsys) -> None:
    prepared, padding = assert_tensor_validity(torch.ones(shape))

    assert prepared.shape[-3:] == (4, 128, 128)
    assert padding == (0, 0, 0, 0)
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "height,width",
    [
        (99, 128),
        (128, 99),
        (99, 99),
        (129, 128),
        (128, 129),
        (256, 256),
        (100, 129),
        (129, 100),
        (1, 1),
    ],
)
@pytest.mark.parametrize("channel_last", [False, True])
def test_tensor_validation_rejects_unsupported_sizes(height, width, channel_last):
    shape = (1, height, width, 4) if channel_last else (1, 4, height, width)
    with pytest.raises(ValueError, match="between 100 and 128") as exc_info:
        assert_tensor_validity(torch.ones(shape))
    message = str(exc_info.value)
    assert "opensr-utils" in message
    assert "tiling, overlap blending, and stitching" in message
    assert "128x128 LR windows" in message


@pytest.mark.parametrize("shape", [(128, 128), (1, 1, 4, 128, 128)])
def test_tensor_validation_rejects_invalid_rank(shape):
    with pytest.raises(ValueError, match="dimensions"):
        assert_tensor_validity(torch.ones(shape))


def test_no_data_mask_is_nearest_neighbor_and_applies_per_band() -> None:
    image = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    mask = create_no_data_mask(image, target_size=4)
    masked = apply_no_data_mask(torch.ones(1, 1, 4, 4), mask)

    assert torch.equal(mask[0, 0, :2, :2], torch.ones(2, 2))
    assert torch.count_nonzero(mask) == 4
    assert torch.equal(masked, 1.0 - mask)


def test_plotting_helpers_create_nonempty_images(tmp_path) -> None:
    example_path = tmp_path / "example.png"
    uncertainty_path = tmp_path / "uncertainty.png"
    plot_example(
        torch.rand(1, 4, 32, 32),
        torch.rand(1, 4, 128, 128),
        out_file=example_path,
    )
    plot_uncertainty(
        torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8),
        out_file=uncertainty_path,
        normalize=False,
    )

    assert example_path.stat().st_size > 0
    assert uncertainty_path.stat().st_size > 0


def test_suppress_stdout_restores_stream(capsys) -> None:
    with suppress_stdout():
        print("hidden")
    print("visible")

    captured = capsys.readouterr()
    assert captured.out == "visible\n"
