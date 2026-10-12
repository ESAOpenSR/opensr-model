from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import torch.nn.functional as F

from opensr_model.srmodel import SRLatentDiffusion, SRLatentDiffusionLightning
from opensr_model.spectral_correction import BackProjectionConfig, BlurDownsample
import opensr_model.srmodel as srmodel_module


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model_stub():
    model = SRLatentDiffusion.__new__(SRLatentDiffusion)
    torch.nn.Module.__init__(model)
    model.config = OmegaConf.create({"denoiser_settings": {
        "sampling_eta": 0., "sampling_temperature": 1., "sampling_steps": 2}})
    model._tensor_encode = MethodType(lambda self, x: x.mul_(.5), model)
    model._tensor_decode = MethodType(lambda self, x, spe_cor=False: F.interpolate(x + .2, scale_factor=4), model)
    sampler = SimpleNamespace(p_sample_ddim=lambda **kw: (kw["x"] + .1, None))
    model._prepare_model = MethodType(lambda self, **kw: (sampler, kw["X"].clone(), np.array([4, 1])), model)
    model.model = SimpleNamespace(num_timesteps=10, parameterization="eps",
        betas=torch.linspace(.001, .02, 10), apply_model=lambda x, t, c: torch.zeros_like(x))
    return model


@pytest.mark.parametrize("sampler", ["ddim", "dpm_solver++"])
@pytest.mark.parametrize("save_iterations", [False, True])
@pytest.mark.parametrize("shape", [(2, 4, 64, 80), (1, 4, 128, 160)])
def test_correction_in_actual_postprocessing_path(sampler, save_iterations, shape):
    if sampler == "dpm_solver++":
        pytest.importorskip("diffusers")
    model = model_stub()
    image = torch.linspace(.05, .4, int(np.prod(shape))).reshape(shape)
    image[0, 0, 10, 12] = 0
    image[0, 1, 11, 12] = torch.nan
    before = image.clone()
    result = model(image, sampler=sampler, spectral_correction="back_projection", save_iterations=save_iterations)
    frames = result if save_iterations else [result]
    clean = torch.nan_to_num(image)
    if save_iterations and shape[-2] < 128:
        clean = F.pad(clean, (24, 24, 32, 32), mode="reflect")
    for frame in frames:
        assert frame.shape == (*clean.shape[:2], clean.shape[-2] * 4, clean.shape[-1] * 4)
        degraded = BlurDownsample().float()(frame)
        torch.testing.assert_close(degraded, clean, atol=1e-6, rtol=1e-6)
        assert torch.count_nonzero(frame[(clean == 0).repeat_interleave(4, -2).repeat_interleave(4, -1)]) == 0
    torch.testing.assert_close(image, before, equal_nan=True)


@pytest.mark.parametrize("explicit, legacy, configured, expected", [
    (None, True, None, "back_projection"), (None, False, None, "none"),
    (None, True, "histogram", "histogram"),
    (None, True, "back_projection", "back_projection"),
    (None, False, "back_projection", "none"),
    ("back_projection", False, "histogram", "back_projection"),
    ("none", True, "back_projection", "none"),
    ("histogram", False, "back_projection", "histogram"),
])
def test_selection_precedence_and_missing_config_defaults(explicit, legacy, configured, expected):
    model = model_stub()
    if configured:
        model.config.spectral_correction = configured
    choice, _ = model._resolve_correction(explicit, legacy, None)
    assert choice == expected


@pytest.mark.parametrize("config_source", ["bundled", "missing_fields"])
def test_default_call_uses_three_dpm_steps_and_back_projection(config_source):
    pytest.importorskip("diffusers")
    model = model_stub()
    if config_source == "bundled":
        model.config = OmegaConf.load(
            Path(srmodel_module.__file__).parent / "configs" / "config_10m.yaml"
        )
    else:
        del model.config.denoiser_settings.sampling_steps
    calls = []
    model.model.apply_model = lambda x, t, c: calls.append(t) or torch.zeros_like(x)
    model._match_output_histograms = MethodType(
        lambda *args: pytest.fail("default inference used histogram matching"), model
    )
    image = torch.linspace(.05, .4, 4 * 64 * 80).reshape(1, 4, 64, 80)

    torch.manual_seed(42)
    result = model(image)
    assert len(calls) == 3
    torch.testing.assert_close(
        F.avg_pool2d(result, 4), image, atol=1e-6, rtol=1e-6
    )

    torch.manual_seed(42)
    explicit = model(image, sampler="dpm_solver++", spectral_correction="back_projection")
    torch.testing.assert_close(result, explicit, atol=0, rtol=0)


def test_config_mapping_and_per_call_dataclass_override():
    model = model_stub()
    model.config.spectral_correction = "back_projection"
    model.config.back_projection = {"blur_sigma": [.8, 1., 1.2, 1.4], "max_iterations": 400}
    _, configured = model._resolve_correction(None, True, None)
    assert configured.blur_sigma == (.8, 1., 1.2, 1.4)
    explicit = BackProjectionConfig(blur_sigma=0., max_iterations=1)
    _, options = model._resolve_correction(None, True, explicit)
    assert options is explicit


@pytest.mark.parametrize("kwargs", [
    {"spectral_correction": "bad"},
    {"spectral_correction": "none", "back_projection": {}},
    {"spectral_correction": "back_projection", "back_projection": {"scale": 2}},
    {"spectral_correction": "back_projection", "back_projection": {"blur_sigma": [1., 2.]}},
])
def test_invalid_options_fail_before_encoding(kwargs):
    model = model_stub()
    model._tensor_encode = MethodType(lambda self, x: pytest.fail("encoded invalid options"), model)
    with pytest.raises(ValueError):
        model(torch.ones(1, 4, 128, 128), **kwargs)


def test_lightning_and_uncertainty_forward_correction_options():
    model = model_stub()
    calls = []
    model.forward = MethodType(lambda self, x, **kwargs: calls.append(kwargs) or x.repeat_interleave(4, -2).repeat_interleave(4, -1), model)
    wrapper = SRLatentDiffusionLightning.__new__(SRLatentDiffusionLightning)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model.eval()
    opts = dict(spectral_correction="back_projection", back_projection={"max_iterations": 2})
    image = torch.ones(1, 4, 16, 20)
    wrapper(image, **opts)
    wrapper.predict_step(image, **opts)
    wrapper.uncertainty_map(image, n_variations=4, **opts)
    assert len(calls) == 6
    assert all(all(call[k] == v for k, v in opts.items()) for call in calls)
