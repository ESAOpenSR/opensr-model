"""Exercise checkpoint transfers and validation with local, tiny model weights."""

import io
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

import opensr_model.srmodel as sr


@pytest.fixture
def model(tiny_model_config):
    return sr.SRLatentDiffusion(OmegaConf.create(tiny_model_config))


def test_download_stream_is_atomic_and_strict_loads_without_perceptual_weights(
    model, tmp_path, monkeypatch
):
    buffer = io.BytesIO()
    state = model.model.state_dict()
    original = {key: tensor.clone() for key, tensor in state.items()}
    state["loss.perceptual.weight"] = torch.ones(1)
    torch.save({"state_dict": state}, buffer)
    payload = buffer.getvalue()
    response = Mock()
    response.headers = {"content-length": str(len(payload))}
    response.iter_content.return_value = [b"", payload[:100], payload[100:]]
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    get = Mock(return_value=response)
    monkeypatch.setattr(sr.requests, "get", get)
    monkeypatch.chdir(tmp_path)
    model.load_pretrained("test.ckpt")
    assert (tmp_path / "test.ckpt").read_bytes() == payload
    assert not (tmp_path / "test.ckpt.tmp").exists()
    response.raise_for_status.assert_called_once()
    assert get.call_args.kwargs == {"stream": True, "timeout": 30}
    assert get.call_args.args[0].endswith("/test.ckpt")
    for key, expected in original.items():
        torch.testing.assert_close(model.model.state_dict()[key], expected)
    model.load_pretrained("test.ckpt")
    assert get.call_count == 1


@pytest.mark.parametrize("during_stream", [False, True])
def test_failed_download_removes_partial_file_and_preserves_weights(
    model, tmp_path, monkeypatch, during_stream
):
    original = next(model.model.parameters()).detach().clone()
    response = Mock()
    response.headers = {}
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)

    def chunks(size):
        yield b"partial data"
        raise OSError("interrupted download")

    if during_stream:
        response.iter_content.side_effect = chunks
    else:
        response.raise_for_status.side_effect = sr.requests.HTTPError("404")
    monkeypatch.setattr(sr.requests, "get", Mock(return_value=response))
    path = tmp_path / "missing.ckpt"
    with pytest.raises(RuntimeError, match="Failed to download") as error:
        model.load_pretrained(str(path))
    assert error.value.__cause__ is not None
    assert not path.exists() and not path.with_suffix(".ckpt.tmp").exists()
    torch.testing.assert_close(next(model.model.parameters()), original)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"state_dict": []},
        {"wrong": {}},
        {"state_dict": {}, "opensr_inference_config": []},
    ],
)
def test_malformed_checkpoint_is_rejected_before_mutation(model, tmp_path, payload):
    path = tmp_path / "malformed.ckpt"
    torch.save(payload, path)
    before = next(model.model.parameters()).detach().clone()
    with pytest.raises(RuntimeError, match="Failed to load pretrained"):
        model.load_pretrained(str(path))
    torch.testing.assert_close(next(model.model.parameters()), before)


@pytest.mark.parametrize(
    "settings,message",
    [
        ({"apply_normalization": 1}, "bool"),
        ({"encode_conditioning": "false"}, "bool"),
        ({"denoiser_settings": []}, "mapping"),
        ({"denoiser_settings": {"timesteps": True}}, "positive int"),
        ({"denoiser_settings": {"timesteps": 0}}, "positive int"),
        ({"denoiser_settings": {"linear_start": True}}, "numeric"),
        ({"denoiser_settings": {"linear_start": float("inf")}}, "finite"),
        ({"denoiser_settings": {"linear_start": 0}}, "positive"),
        ({"denoiser_settings": {"linear_start": 1}}, "below 1"),
        ({"denoiser_settings": {"linear_end": 1}}, "below 1"),
        (
            {"denoiser_settings": {"linear_start": 0.5, "linear_end": 0.1}},
            "linear_start <= linear_end",
        ),
        ({"denoiser_settings": {"parameterization": "v"}}, "parameterization"),
        ({"denoiser_settings": {"sampler": "invalid"}}, "sampler"),
        ({"denoiser_settings": {"sampling_steps": False}}, "positive int"),
        ({"denoiser_settings": {"sampling_steps": 10}}, "smaller than"),
        ({"denoiser_settings": {"sampling_steps": 3}}, "divide timesteps"),
        ({"denoiser_settings": {"sampling_eta": -1}}, "non-negative"),
        ({"denoiser_settings": {"sampling_temperature": "warm"}}, "numeric"),
        ({"denoiser_settings": {"timesteps": 10}}, "no 'betas' tensor"),
    ],
)
def test_invalid_inference_metadata_is_transactional(model, settings, message):
    before = OmegaConf.to_container(model.config)
    with pytest.raises(RuntimeError, match=message):
        model._prepare_checkpoint_inference_config(settings, {})
    assert OmegaConf.to_container(model.config) == before


def test_metadata_can_update_plain_attributes_without_changing_unspecified_values(
    model,
):
    target = SimpleNamespace(old=7)
    model._set_config_value(target, "new", 3)
    assert target.new == 3 and target.old == 7
    assert model._prepare_checkpoint_inference_config(
        {"apply_normalization": False}, {}
    ) == {"apply_normalization": False}
    nested = {"denoiser_settings": {}}
    assert (
        model._checkpoint_inference_config({"opensr_inference_config": nested})
        is nested
    )


@pytest.mark.parametrize("encode", [False, True])
def test_tensor_encoding_and_decoding_use_real_autoencoder(model, encode):
    model.encode_conditioning = encode
    image = torch.rand(1, 4, 4, 4)
    encoded = model._tensor_encode(image)
    torch.testing.assert_close(model._X, image)
    assert model._X.data_ptr() != image.data_ptr()
    if encode:
        assert encoded.shape == (1, 2, 8, 8)
        decoded = model._tensor_decode(encoded, spe_cor=False)
        assert decoded.shape == (1, 4, 16, 16)
        assert decoded.min() >= 0
        corrected = model._tensor_decode(encoded, spe_cor=True)
        assert corrected.shape == decoded.shape and torch.isfinite(corrected).all()
    else:
        torch.testing.assert_close(encoded, image)


def test_histogram_matching_rejects_mismatched_batches_and_channels(model):
    with pytest.raises(ValueError, match="BCHW"):
        model._match_output_histograms(torch.ones(4, 8, 8), torch.ones(4, 2, 2))
    with pytest.raises(ValueError, match="equal batch"):
        model._match_output_histograms(torch.ones(2, 4, 8, 8), torch.ones(1, 4, 2, 2))


def test_lightning_wrapper_constructs_and_loads_checkpoint(tiny_model_config, tmp_path):
    config = OmegaConf.create(tiny_model_config)
    wrapper = sr.SRLatentDiffusionLightning(config)
    path = tmp_path / "native.ckpt"
    torch.save({"state_dict": wrapper.model.model.state_dict()}, path)
    wrapper.load_pretrained(str(path))
    assert isinstance(wrapper.model, sr.SRLatentDiffusion)
