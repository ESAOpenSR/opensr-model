from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from opensr_model.spectral_correction import (
    BackProjectionConfig,
    BlurDownsample,
    back_projection,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def numpy_reference(source, observed, sigmas, steps, tolerance):
    """Independent SciPy implementation, same as the experimental benchmark."""
    x = source.copy()

    def blur(a):
        return np.stack(
            [
                gaussian_filter(b, s, mode="wrap") if s else b.copy()
                for b, s in zip(a, sigmas)
            ]
        )

    for step in range(steps + 1):
        degraded = (
            blur(x).reshape(4, observed.shape[1], 4, observed.shape[2], 4).mean((2, 4))
        )
        residual = observed - degraded
        if np.sqrt(np.mean(residual**2)) <= tolerance or step == steps:
            break
        x += blur(residual.repeat(4, 1).repeat(4, 2))
    return x


@pytest.mark.parametrize("sigmas", [(0.0,) * 4, (0.8, 1.0, 1.2, 1.4)])
def test_torch_matches_scipy_reference_and_monotone_residuals(sigmas):
    rng = np.random.default_rng(42)
    source = rng.uniform(0.1, 0.6, (4, 24, 32))
    observed = rng.uniform(0.1, 0.6, (4, 6, 8))
    config = BackProjectionConfig(blur_sigma=sigmas, max_iterations=400, tolerance=1e-9)
    expected = numpy_reference(source, observed, sigmas, 400, 1e-9)
    actual, info = back_projection(
        torch.from_numpy(source), torch.from_numpy(observed), config, return_info=True
    )
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-12)
    assert torch.all(torch.diff(info["residual_history"], dim=0) <= 1e-12)
    assert info["converged"].all()


@pytest.mark.parametrize("sigmas", [0.0, (0.8, 1.0, 1.2, 1.4)])
def test_adjoint_and_constants_on_rectangular_and_tiny_images(sigmas):
    op = BlurDownsample(4, scale=2, blur_sigma=sigmas).double()
    x = torch.randn(2, 4, 4, 6, dtype=torch.float64)
    r = torch.randn(2, 4, 2, 3, dtype=torch.float64)
    torch.testing.assert_close(
        (op(x) * r).sum(), (x * op.adjoint(r)).sum(), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(op(torch.ones_like(x)), torch.ones_like(r))


def test_projection_matches_dense_pseudoinverse():
    op = BlurDownsample(1, scale=2, blur_sigma=0.8).double()
    x, y = torch.randn(1, 1, 8, 8, dtype=torch.float64), torch.randn(
        1, 1, 4, 4, dtype=torch.float64
    )
    matrix = (
        op(torch.eye(64, dtype=torch.float64).reshape(64, 1, 8, 8)).reshape(64, 16).T
    )
    expected = x.flatten() + torch.linalg.pinv(matrix) @ (
        y.flatten() - matrix @ x.flatten()
    )
    result = back_projection(
        x,
        y,
        BackProjectionConfig(
            scale=2, blur_sigma=0.8, max_iterations=3000, tolerance=1e-11
        ),
    )
    torch.testing.assert_close(result.flatten(), expected, atol=1e-9, rtol=1e-9)


def test_block_means_preserve_detail_and_inputs():
    x = torch.randn(2, 4, 16, 24, dtype=torch.float64)
    y = torch.rand(2, 4, 4, 6, dtype=torch.float64)
    before = x.clone()
    op = BlurDownsample().double()
    result, info = back_projection(x, y, return_info=True)
    expected = x + (y - op(x)).repeat_interleave(4, -2).repeat_interleave(4, -1)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(op(result), y)
    assert torch.equal(x, before)
    assert torch.equal(info["iterations"], torch.ones(2, dtype=torch.long))
    assert result.min() < 0  # Unconstrained projection deliberately does not clip.


def test_batch_stopping_is_independent_of_other_samples():
    torch.manual_seed(8)
    op = BlurDownsample(4, blur_sigma=1.0).double()
    source = torch.rand(2, 4, 16, 20, dtype=torch.float64)
    y = op(source)
    y[1] += torch.randn_like(y[1]) * 0.01
    config = BackProjectionConfig(blur_sigma=1.0, max_iterations=400)
    combined, info = back_projection(source, y, config, return_info=True)
    separate = back_projection(source[1], y[1], config)
    assert info["iterations"][0] == 0 and info["iterations"][1] > 1
    assert torch.equal(combined[0], source[0])
    torch.testing.assert_close(combined[1], separate, atol=1e-12, rtol=1e-12)


def test_nodata_masks_hold_output_at_zero_and_ignore_invalid_measurements():
    x = torch.full((2, 4, 16, 20), 0.2)
    y = torch.full((2, 4, 4, 5), 0.3)
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[0, :, 1, 1] = False
    valid[1] = False
    y[~valid] = torch.nan
    support = valid.repeat_interleave(4, -2).repeat_interleave(4, -1)
    result, info = back_projection(
        x, y, valid_mask=valid, output_mask=support, return_info=True
    )
    assert torch.all(result[~support] == 0)
    torch.testing.assert_close(BlurDownsample().float()(result)[valid], y[valid])
    assert info["converged"].tolist() == [True, False]
    assert info["iterations"].tolist() == [1, 0]


def test_blurred_masked_projection_preserves_support_and_converges():
    torch.manual_seed(9)
    support = torch.ones(1, 4, 16, 20, dtype=torch.bool)
    support[:, :, :4, :4] = False
    truth = torch.rand(1, 4, 16, 20, dtype=torch.float64) * support
    op = BlurDownsample(4, blur_sigma=1.0).double()
    y = op(truth)
    valid = torch.ones_like(y, dtype=torch.bool)
    valid[:, :, 0, 0] = False
    result, info = back_projection(
        truth + 0.04,
        y,
        BackProjectionConfig(blur_sigma=1.0, max_iterations=400),
        valid_mask=valid,
        output_mask=support,
        return_info=True,
    )
    assert torch.count_nonzero(result[~support]) == 0
    assert info["converged"].all()
    assert torch.all(torch.diff(info["residual_history"], dim=0) <= 1e-12)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_dtype_promotion_device_and_autocast(dtype):
    source = torch.full((4, 16, 20), 0.2, dtype=dtype)
    observed = torch.full((4, 4, 5), 0.3, dtype=dtype)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = back_projection(source, observed)
    assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    assert result.device == source.device
    torch.testing.assert_close(
        result,
        observed.to(result.dtype).repeat_interleave(4, -2).repeat_interleave(4, -1),
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": 0},
        {"max_iterations": -1},
        {"max_iterations": True},
        {"tolerance": float("nan")},
        {"relaxation": 1.1},
        {"blur_sigma": -1.0},
        {"blur_sigma": (1.0, float("inf"))},
    ],
)
def test_bad_options_fail(kwargs):
    with pytest.raises(ValueError):
        BackProjectionConfig(**kwargs)


def test_bad_shapes_bands_masks_and_nonfinite_values():
    x, y = torch.ones(1, 4, 16, 20), torch.ones(1, 4, 4, 5)
    with pytest.raises(ValueError, match="spatial shape"):
        back_projection(x, y[..., :4])
    with pytest.raises(ValueError, match="one value per"):
        back_projection(x, y, BackProjectionConfig(blur_sigma=(1.0, 2.0)))
    with pytest.raises(ValueError, match="boolean"):
        back_projection(x, y, valid_mask=torch.ones_like(y))
    with pytest.raises(ValueError, match="finite"):
        back_projection(x * float("nan"), y)
    with pytest.raises(ValueError, match="floating"):
        back_projection(x.int(), y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_matches_cpu_and_stays_on_device():
    x, y = torch.rand(1, 4, 16, 20), torch.rand(1, 4, 4, 5)
    config = BackProjectionConfig(blur_sigma=1.0, max_iterations=400)
    cpu = back_projection(x, y, config)
    gpu = back_projection(x.cuda(), y.cuda(), config)
    assert gpu.is_cuda
    torch.testing.assert_close(cpu, gpu.cpu(), atol=2e-5, rtol=2e-5)
