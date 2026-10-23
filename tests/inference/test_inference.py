from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

import opensr_model.srmodel as srmodel_module
from opensr_model.srmodel import SRLatentDiffusion, SRLatentDiffusionLightning


class _Sampler:
    def __init__(self) -> None:
        self.calls = []

    def p_sample_ddim(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["x"] + 1, torch.zeros_like(kwargs["x"])


def _stub_inference_model() -> tuple[SRLatentDiffusion, _Sampler]:
    model = SRLatentDiffusion.__new__(SRLatentDiffusion)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        spectral_correction="histogram",
        denoiser_settings=SimpleNamespace(
            sampler="ddim",
            sampling_eta=0.25,
            sampling_temperature=0.75,
            sampling_steps=2,
        ),
    )
    model.device = torch.device("cpu")
    sampler = _Sampler()
    model._tensor_encode = MethodType(lambda self, value: value + 10, model)
    model._prepare_model = MethodType(
        lambda self, X, eta, custom_steps, verbose: (
            sampler,
            torch.zeros_like(X),
            np.array([9, 4]),
        ),
        model,
    )
    model._tensor_decode = MethodType(
        lambda self, latent, spe_cor=True: latent * 2, model
    )
    return model, sampler


def test_forward_uses_config_defaults_and_runs_each_reverse_step(monkeypatch) -> None:
    model, sampler = _stub_inference_model()
    utility_calls = []
    monkeypatch.setattr(
        srmodel_module,
        "assert_tensor_validity",
        lambda value: (value, (1, 2, 3, 4)),
    )
    monkeypatch.setattr(
        srmodel_module,
        "create_no_data_mask",
        lambda value, target_size: torch.zeros_like(value),
    )
    monkeypatch.setattr(
        srmodel_module,
        "apply_no_data_mask",
        lambda value, mask: utility_calls.append("mask") or value,
    )
    monkeypatch.setattr(
        srmodel_module,
        "revert_padding",
        lambda value, padding: utility_calls.append(("padding", padding)) or value,
    )

    result = model(torch.ones(1, 4, 8, 8), histogram_matching=False)

    assert torch.equal(result, torch.full((1, 4, 8, 8), 4.0))
    assert [call["t"] for call in sampler.calls] == [9, 4]
    assert [call["index"] for call in sampler.calls] == [1, 0]
    assert all(call["temperature"] == 0.75 for call in sampler.calls)
    assert utility_calls == ["mask", ("padding", (1, 2, 3, 4))]


def test_forward_histogram_matches_only_after_final_postprocessing(monkeypatch) -> None:
    model, _ = _stub_inference_model()
    decode_flags = []
    matched_sources = []
    model._tensor_decode = MethodType(
        lambda self, latent, spe_cor=True: decode_flags.append(spe_cor) or latent * 2,
        model,
    )
    model.hq_histogram_matching = MethodType(
        lambda self, source, reference: (
            matched_sources.append((source.clone(), reference.clone())) or source + 1000
        ),
        model,
    )
    monkeypatch.setattr(
        srmodel_module,
        "assert_tensor_validity",
        lambda value: (value, (0, 0, 0, 0)),
    )
    monkeypatch.setattr(
        srmodel_module,
        "create_no_data_mask",
        lambda value, target_size: torch.zeros_like(value),
    )
    monkeypatch.setattr(
        srmodel_module, "apply_no_data_mask", lambda value, mask: value + 10
    )
    monkeypatch.setattr(
        srmodel_module, "revert_padding", lambda value, padding: value + 100
    )

    lr = torch.ones(1, 4, 8, 8)
    result = model(lr)

    assert decode_flags == [False]
    assert len(matched_sources) == 4
    assert all(
        torch.equal(source, torch.full_like(source, 114))
        for source, _ in matched_sources
    )
    assert all(
        torch.equal(reference, lr[0, band])
        for band, (_, reference) in enumerate(matched_sources)
    )
    assert torch.equal(result, torch.full_like(result, 1114))


def test_forward_sanitizes_input_and_can_return_decoded_iterations(monkeypatch) -> None:
    model, sampler = _stub_inference_model()
    captured = {}

    def validate(value):
        captured["value"] = value.clone()
        return value, (0, 0, 0, 0)

    monkeypatch.setattr(srmodel_module, "assert_tensor_validity", validate)
    monkeypatch.setattr(
        srmodel_module, "create_no_data_mask", lambda value, target_size: None
    )
    image = torch.tensor([[[[float("nan"), float("inf")]]]]).expand(1, 4, 1, 2)

    iterations = model(
        image,
        sampling_eta=0.0,
        sampling_steps=2,
        sampling_temperature=0.0,
        histogram_matching=False,
        save_iterations=True,
    )

    assert torch.count_nonzero(captured["value"]) == 0
    assert len(iterations) == 2
    assert torch.equal(iterations[0], torch.full_like(image, 2.0))
    assert torch.equal(iterations[1], torch.full_like(image, 4.0))
    assert all(call["temperature"] == 0.0 for call in sampler.calls)


def test_histogram_matching_supports_images_and_rejects_invalid_rank() -> None:
    model = SRLatentDiffusion.__new__(SRLatentDiffusion)
    source = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    reference = torch.tensor([[2.0, 2.0], [4.0, 4.0]])

    matched = model.hq_histogram_matching(source, reference)

    assert matched.device == source.device
    assert matched.shape == source.shape
    assert matched.min() >= reference.min()
    assert matched.max() <= reference.max()
    with pytest.raises(ValueError, match="2 or 3 dimensions"):
        model.hq_histogram_matching(torch.zeros(2), torch.zeros(2))


@pytest.mark.parametrize("sampling_steps", [None, 3])
def test_uncertainty_map_returns_twice_channel_mean_standard_deviation(
    monkeypatch,
    sampling_steps,
) -> None:
    model = SRLatentDiffusion.__new__(SRLatentDiffusion)
    torch.nn.Module.__init__(model)
    draw = {"value": 0}
    sampling_calls = []

    def forward(self, image, **kwargs):
        sampling_calls.append(kwargs)
        value = draw["value"] % 4
        draw["value"] += 1
        return torch.full((1, 4, 3, 3), float(value))

    model.forward = MethodType(forward, model)
    monkeypatch.setattr(
        srmodel_module.random,
        "sample",
        lambda population, count: list(range(count)),
    )

    kwargs = {} if sampling_steps is None else {"sampling_steps": sampling_steps}
    result = model.uncertainty_map(
        torch.zeros(2, 4, 1, 1),
        n_variations=4,
        sampler="dpm_solver++",
        **kwargs,
    )

    expected = 2 * torch.tensor([0.0, 1.0, 2.0, 3.0]).std()
    assert result.shape == (2, 1, 3, 3)
    assert torch.allclose(result, torch.full_like(result, expected))
    assert all(call["sampler"] == "dpm_solver++" for call in sampling_calls)
    assert all(call["sampling_steps"] == sampling_steps for call in sampling_calls)
    with pytest.raises(AssertionError, match="greater than 3"):
        model.uncertainty_map(torch.zeros(1, 4, 1, 1), n_variations=3)


@pytest.mark.parametrize("save_iterations", [False, True])
@pytest.mark.parametrize(
    "selection",
    [
        "configured_count",
        "config_sampler",
        "call_sampler",
        "call_count",
        "missing_count",
    ],
)
def test_dpm_forward_uses_config_and_resets_history(
    monkeypatch, save_iterations, selection
):
    pytest.importorskip("diffusers")
    model, ddim = _stub_inference_model()
    model.config.denoiser_settings.sampler = "dpm_solver++"
    model.config.denoiser_settings.sampling_steps = None
    kwargs = {}
    expected_steps = 5 if selection == "configured_count" else 3
    if selection == "configured_count":
        model.config.denoiser_settings.sampling_steps = expected_steps
    elif selection == "call_sampler":
        model.config.denoiser_settings.sampler = "ddim"
        kwargs["sampler"] = "dpm_solver++"
    elif selection == "call_count":
        model.config.denoiser_settings.sampling_steps = 2
        kwargs["sampling_steps"] = 3
    elif selection == "missing_count":
        del model.config.denoiser_settings.sampling_steps
    calls = []
    model.model = SimpleNamespace(
        num_timesteps=10,
        parameterization="eps",
        betas=torch.linspace(0.001, 0.02, 10),
        apply_model=lambda x, t, c: (
            calls.append((t.clone(), c.clone())) or torch.zeros_like(x)
        ),
    )
    monkeypatch.setattr(
        srmodel_module, "assert_tensor_validity", lambda x: (x, (0, 0, 0, 0))
    )
    monkeypatch.setattr(
        srmodel_module, "create_no_data_mask", lambda x, target_size: None
    )
    monkeypatch.setattr(srmodel_module, "apply_no_data_mask", lambda x, mask: x + 10)
    monkeypatch.setattr(srmodel_module, "revert_padding", lambda x, padding: x + 100)
    model._match_output_histograms = MethodType(lambda self, x, ref: x + 1000, model)
    x = torch.ones(2, 4, 8, 8)

    torch.manual_seed(42)
    first = model(x, save_iterations=save_iterations, **kwargs)
    torch.manual_seed(42)
    second = model(x, save_iterations=save_iterations, **kwargs)
    assert len(calls) == 2 * expected_steps
    assert all(torch.equal(c, x + 10) for _, c in calls)
    assert all(t.shape == (2,) for t, _ in calls)
    assert not ddim.calls
    if save_iterations:
        assert len(first) == expected_steps
        assert all(torch.equal(a, b) for a, b in zip(first, second))
        final = first[-1]
        postprocess_offset = 1000
    else:
        assert torch.equal(first, second)
        final = first
        postprocess_offset = 1110
    torch.manual_seed(42)
    expected = 2 * torch.randn(x.shape) / (1 - model.model.betas).prod().sqrt()
    torch.testing.assert_close(final, expected + postprocess_offset)

    # An explicit argument overrides the config and keeps DDIM's noise defaults.
    model(x, sampler="ddim", sampling_steps=2)
    assert len(ddim.calls) == 2
    assert all(call["temperature"] == 0.75 for call in ddim.calls)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"sampler": "unknown"}, "sampler must"),
        ({"sampler": "dpm_solver++", "sampling_eta": 0.95}, "sampling_eta"),
        (
            {"sampler": "dpm_solver++", "sampling_temperature": 0},
            "sampling_temperature",
        ),
    ],
)
def test_invalid_sampler_settings_fail_before_encoding(kwargs, message):
    model, _ = _stub_inference_model()
    with pytest.raises(ValueError, match=message):
        model(torch.ones(1, 4, 8, 8), **kwargs)


def test_lightning_wrapper_forwards_sampling_arguments():
    wrapper = SRLatentDiffusionLightning.__new__(SRLatentDiffusionLightning)
    torch.nn.Module.__init__(wrapper)
    calls = []
    uncertainty_counts = []

    class Recorder(torch.nn.Module):
        def forward(self, x, **kwargs):
            calls.append(kwargs)
            return x

        def uncertainty_map(self, x, *args, **kwargs):
            calls.append(kwargs)
            uncertainty_counts.append(args[1])
            return x

    wrapper.model = Recorder().eval()
    x = torch.ones(1, 4, 8, 8)
    wrapper(x, sampler="dpm_solver++", sampling_steps=20)
    wrapper.predict_step(x, dataloader_idx=1, sampler="dpm_solver++", sampling_steps=20)
    wrapper.uncertainty_map(x, sampler="dpm_solver++")
    assert calls[:2] == [{"sampler": "dpm_solver++", "sampling_steps": 20}] * 2
    assert calls[2] == {"sampler": "dpm_solver++"}
    assert uncertainty_counts == [None]
    wrapper.uncertainty_map(x, sampling_steps=5, custom_steps=7, sampler="dpm_solver++")
    assert uncertainty_counts[-1] == 7


@pytest.mark.parametrize(
    "sampler, expected_steps", [("ddim", 100), ("dpm_solver++", 3)]
)
def test_bundled_config_uses_sampler_specific_default_steps(
    monkeypatch, sampler, expected_steps
):
    from pathlib import Path
    from omegaconf import OmegaConf

    model, ddim = _stub_inference_model()
    model.config = OmegaConf.load(
        Path(srmodel_module.__file__).parent / "configs" / "config_10m.yaml"
    )
    model.model = SimpleNamespace()
    observed = []

    def prepare(self, X, eta, custom_steps, verbose):
        observed.append(custom_steps)
        return ddim, torch.zeros_like(X), np.arange(custom_steps)

    class DPM:
        def __init__(self, model, steps):
            observed.append(steps)
            self.timesteps = list(range(steps))

        def step(self, latent, conditioning, step):
            return latent

    model._prepare_model = MethodType(prepare, model)
    monkeypatch.setattr(srmodel_module, "DPMSolverPlusPlusSampler", DPM)
    monkeypatch.setattr(
        srmodel_module, "assert_tensor_validity", lambda x: (x, (0, 0, 0, 0))
    )
    monkeypatch.setattr(
        srmodel_module, "create_no_data_mask", lambda x, target_size: None
    )
    monkeypatch.setattr(srmodel_module, "apply_no_data_mask", lambda x, mask: x)
    monkeypatch.setattr(srmodel_module, "revert_padding", lambda x, padding: x)
    model(torch.ones(1, 4, 8, 8), sampler=sampler, histogram_matching=False)
    assert observed == [expected_steps]


@pytest.mark.parametrize("steps", [0, -1, 3, 10, 11, True, 2.5])
def test_ddim_rejects_invalid_steps_when_switching_from_dpm(steps):
    model = SRLatentDiffusion.__new__(SRLatentDiffusion)
    torch.nn.Module.__init__(model)
    model.model = SimpleNamespace(num_timesteps=10)
    with pytest.raises(ValueError, match="DDIM sampling_steps"):
        model._prepare_model(torch.zeros(1, 4, 8, 8), custom_steps=steps)


@pytest.mark.parametrize("shape", [(1, 4, 99, 128), (1, 4, 128, 129), (4, 256, 256)])
def test_forward_rejects_unsupported_patch_before_encoding(shape):
    model, sampler = _stub_inference_model()
    model._tensor_encode = MethodType(
        lambda self, value: pytest.fail("encoded unsupported input"), model
    )
    with pytest.raises(ValueError, match="trained for 128x128 -> 512x512") as exc_info:
        model(torch.ones(shape))
    assert "opensr-utils" in str(exc_info.value)
    assert sampler.calls == []


@pytest.mark.parametrize(
    "height,width", [(100, 100), (100, 128), (128, 100), (127, 101), (128, 128)]
)
def test_forward_pads_supported_patches_and_crops_final_output(height, width, capsys):
    model, _ = _stub_inference_model()
    encoded_shapes = []

    def encode(self, value):
        encoded_shapes.append(tuple(value.shape))
        return value

    model._tensor_encode = MethodType(encode, model)
    model._tensor_decode = MethodType(
        lambda self, latent, spe_cor=False: torch.nn.functional.interpolate(
            latent, scale_factor=4, mode="nearest"
        ),
        model,
    )
    result = model(torch.ones(1, 4, height, width), histogram_matching=False)

    assert encoded_shapes == [(1, 4, 128, 128)]
    assert result.shape == (1, 4, height * 4, width * 4)
    assert torch.equal(result, torch.full_like(result, 2.0))
    message = capsys.readouterr().out
    if height < 128 or width < 128:
        assert f"reflect-padding {height}x{width} LR input to 128x128" in message
    else:
        assert message == ""


@pytest.mark.parametrize("entry_point", ["forward", "predict_step"])
@pytest.mark.parametrize("height,width", [(99, 128), (128, 129)])
def test_lightning_rejects_unsupported_patches_with_tiling_guidance(
    entry_point, height, width
):
    model, sampler = _stub_inference_model()
    wrapper = SRLatentDiffusionLightning.__new__(SRLatentDiffusionLightning)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model.eval()

    with pytest.raises(ValueError, match="opensr-utils") as exc_info:
        getattr(wrapper, entry_point)(torch.ones(1, 4, height, width))
    assert "tiling, overlap blending, and stitching" in str(exc_info.value)
    assert sampler.calls == []


@pytest.mark.parametrize("encode_conditioning", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        "meta",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_forward_moves_cpu_input_before_encoding_and_postprocessing(
    device, encode_conditioning
):
    model, sampler = _stub_inference_model()
    encoded_devices = []

    class ConditioningEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 4, 1)

        def encode(self, value):
            encoded_devices.append(value.device.type)
            # A real convolution reproduces the original CPU/CUDA failure if
            # the input is not moved before conditioning is encoded.
            latent = self.conv(torch.nn.functional.avg_pool2d(value, 4))
            return SimpleNamespace(sample=lambda: latent)

    model.model = torch.nn.Module()
    model.model.first_stage_model = ConditioningEncoder()
    model.to(device)
    # Constructor device attributes can become stale after a later .to(...).
    model.device = "meta" if device == "cpu" else "cpu"
    model.encode_conditioning = encode_conditioning
    model.linear_transform = lambda value, stage: value
    model._tensor_encode = MethodType(SRLatentDiffusion._tensor_encode, model)
    model._tensor_decode = MethodType(
        lambda self, latent, spe_cor=False: torch.nn.functional.interpolate(
            latent, scale_factor=4, mode="nearest"
        ),
        model,
    )
    reference_devices = []

    def correct(self, output, reference, correction, options):
        reference_devices.append(reference.device.type)
        assert output.device == reference.device
        return output

    model._correct_output = MethodType(correct, model)
    image = torch.ones(1, 4, 100, 100)
    result = model(image, histogram_matching=False)

    assert image.device.type == "cpu"
    assert result.device.type == device
    assert result.shape == (1, 4, 400, 400)
    assert model._X.device.type == device
    assert reference_devices == [device]
    assert encoded_devices == ([device] if encode_conditioning else [])
    assert all(call["x"].device.type == device for call in sampler.calls)
