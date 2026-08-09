from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from opensr_model.srmodel import SRLatentDiffusion
from opensr_model.utils import linear_transform_4b


def _inference_config(model: SRLatentDiffusion) -> dict[str, object]:
    betas = model.model.betas.detach().cpu()
    return {
        "apply_normalization": True,
        "encode_conditioning": False,
        "denoiser_settings": {
            "timesteps": int(model.model.num_timesteps),
            "linear_start": float(betas[0]),
            "linear_end": float(betas[-1]),
            "parameterization": "eps",
            "sampling_steps": 6,
            "sampling_eta": 0.75,
            "sampling_temperature": 0.8,
        },
    }


def test_load_pretrained_applies_native_inference_metadata(
    tmp_path, tiny_model_config
) -> None:
    source_config = deepcopy(tiny_model_config)
    source_config["denoiser_settings"].update(
        {"timesteps": 12, "linear_start": 2.0e-4, "linear_end": 1.2e-2}
    )
    source = SRLatentDiffusion(OmegaConf.create(source_config))
    inference_config = _inference_config(source)
    checkpoint = tmp_path / "native.ckpt"
    torch.save(
        {
            "state_dict": source.model.state_dict(),
            "metadata": {"opensr_inference_config": inference_config},
        },
        checkpoint,
    )

    restored_config = deepcopy(tiny_model_config)
    restored_config["denoiser_settings"]["parameterization"] = "x0"
    restored = SRLatentDiffusion(OmegaConf.create(restored_config))
    restored.load_pretrained(str(checkpoint))

    assert restored.model.num_timesteps == 12
    assert restored.model.parameterization == "eps"
    assert restored.config.denoiser_settings.parameterization == "eps"
    assert restored.config.denoiser_settings.timesteps == 12
    assert restored.config.denoiser_settings.linear_start == pytest.approx(
        inference_config["denoiser_settings"]["linear_start"]
    )
    assert restored.config.denoiser_settings.linear_end == pytest.approx(
        inference_config["denoiser_settings"]["linear_end"]
    )
    assert restored.config.denoiser_settings.sampling_steps == 6
    assert restored.config.denoiser_settings.sampling_eta == pytest.approx(0.75)
    assert restored.config.denoiser_settings.sampling_temperature == pytest.approx(0.8)
    assert restored.config.apply_normalization is True
    assert restored.config.encode_conditioning is False
    assert restored.encode_conditioning is False
    assert restored.linear_transform is linear_transform_4b
    for key, value in source.model.state_dict().items():
        assert torch.equal(restored.model.state_dict()[key], value)


def test_load_pretrained_without_metadata_keeps_legacy_config(
    tmp_path, tiny_model_config
) -> None:
    config = OmegaConf.create(tiny_model_config)
    model = SRLatentDiffusion(config)
    checkpoint = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": model.model.state_dict()}, checkpoint)
    before = OmegaConf.to_container(config, resolve=True)

    model.load_pretrained(str(checkpoint))

    assert OmegaConf.to_container(config, resolve=True) == before
    assert (
        model.model.num_timesteps == tiny_model_config["denoiser_settings"]["timesteps"]
    )
    assert model.model.parameterization == "eps"


def test_load_pretrained_rejects_metadata_timestep_state_mismatch(
    tmp_path, tiny_model_config
) -> None:
    model = SRLatentDiffusion(OmegaConf.create(tiny_model_config))
    inference_config = _inference_config(model)
    inference_config["denoiser_settings"]["timesteps"] = 9
    inference_config["denoiser_settings"]["sampling_steps"] = 1
    checkpoint = tmp_path / "mismatched.ckpt"
    torch.save(
        {
            "state_dict": model.model.state_dict(),
            "opensr_inference_config": inference_config,
        },
        checkpoint,
    )

    with pytest.raises(RuntimeError, match="timesteps=9.*'betas'.*10 values"):
        model.load_pretrained(str(checkpoint))


def test_malformed_sampling_metadata_does_not_partially_mutate_model(
    tmp_path, tiny_model_config
) -> None:
    config = OmegaConf.create(tiny_model_config)
    model = SRLatentDiffusion(config)
    inference_config = _inference_config(model)
    # Ten native timesteps cannot produce exactly six uniform DDIM steps.
    inference_config["denoiser_settings"]["sampling_steps"] = 6
    checkpoint = tmp_path / "bad-sampling.ckpt"
    torch.save(
        {
            "state_dict": model.model.state_dict(),
            "metadata": {"opensr_inference_config": inference_config},
        },
        checkpoint,
    )
    config_before = OmegaConf.to_container(config, resolve=True)
    betas_before = model.model.betas.detach().clone()
    parameterization_before = model.model.parameterization

    with pytest.raises(RuntimeError, match="sampling_steps must divide timesteps"):
        model.load_pretrained(str(checkpoint))

    assert OmegaConf.to_container(config, resolve=True) == config_before
    assert model.model.num_timesteps == 10
    assert model.model.parameterization == parameterization_before
    assert torch.equal(model.model.betas, betas_before)


def test_load_pretrained_rejects_unsupported_x0_metadata(
    tmp_path, tiny_model_config
) -> None:
    config = OmegaConf.create(tiny_model_config)
    model = SRLatentDiffusion(config)
    inference_config = _inference_config(model)
    inference_config["denoiser_settings"]["parameterization"] = "x0"
    inference_config["denoiser_settings"]["sampling_steps"] = 5
    checkpoint = tmp_path / "x0.ckpt"
    torch.save(
        {
            "state_dict": model.model.state_dict(),
            "opensr_inference_config": inference_config,
        },
        checkpoint,
    )
    config_before = OmegaConf.to_container(config, resolve=True)

    with pytest.raises(
        RuntimeError, match="requires checkpoint parameterization='eps'"
    ):
        model.load_pretrained(str(checkpoint))

    assert OmegaConf.to_container(config, resolve=True) == config_before
    assert model.model.parameterization == "eps"
