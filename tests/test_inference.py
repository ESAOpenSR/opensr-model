from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

import opensr_model.srmodel as srmodel_module
from opensr_model.srmodel import SRLatentDiffusion


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
        denoiser_settings=SimpleNamespace(
            sampling_eta=0.25, sampling_temperature=0.75, sampling_steps=2
        )
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
        lambda self, source, reference: matched_sources.append(
            (source.clone(), reference.clone())
        )
        or source + 1000,
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


def test_uncertainty_map_returns_twice_channel_mean_standard_deviation(
    monkeypatch,
) -> None:
    model = SRLatentDiffusion.__new__(SRLatentDiffusion)
    torch.nn.Module.__init__(model)
    draw = {"value": 0}

    def forward(self, image, **kwargs):
        value = draw["value"] % 4
        draw["value"] += 1
        return torch.full((1, 4, 3, 3), float(value))

    model.forward = MethodType(forward, model)
    monkeypatch.setattr(
        srmodel_module.random,
        "sample",
        lambda population, count: list(range(count)),
    )

    result = model.uncertainty_map(
        torch.zeros(2, 4, 1, 1), n_variations=4, sampling_steps=3
    )

    expected = 2 * torch.tensor([0.0, 1.0, 2.0, 3.0]).std()
    assert result.shape == (2, 1, 3, 3)
    assert torch.allclose(result, torch.full_like(result, expected))
    with pytest.raises(AssertionError, match="greater than 3"):
        model.uncertainty_map(torch.zeros(1, 4, 1, 1), n_variations=3)
