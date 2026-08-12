from __future__ import annotations

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


def test_tensor_validation_sanitizes_pads_and_padding_is_reversible() -> None:
    image = torch.ones(4, 64, 80)
    image[0, 0, 0] = torch.nan
    image[1, 0, 0] = torch.inf
    prepared, padding = assert_tensor_validity(image)

    assert prepared.shape == (1, 4, 128, 128)
    assert torch.isfinite(prepared).all()
    assert padding == (24, 24, 32, 32)
    super_resolved = torch.arange(4 * 512 * 512).reshape(1, 4, 512, 512)
    reverted = revert_padding(super_resolved, padding)
    assert reverted.shape == (1, 4, 256, 320)
    assert torch.equal(reverted, super_resolved[:, :, 128:384, 96:416])


def test_tensor_validation_accepts_channel_last_and_large_images() -> None:
    channel_last = torch.ones(1, 128, 160, 4)
    prepared, padding = assert_tensor_validity(channel_last)

    assert prepared.shape == (1, 4, 160, 128)
    assert padding == (0, 0, 0, 0)


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
