"""Shared settings for the small CPU models used by the test suite."""

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_threaded_torch():
    """Avoid thread-pool overhead on tiny tensors during every commit."""
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def tiny_model_config() -> dict:
    first_stage = {
        "embed_dim": 2,
        "double_z": True,
        "z_channels": 2,
        "resolution": 16,
        "in_channels": 4,
        "out_ch": 4,
        "ch": 32,
        "ch_mult": [1, 2],
        "num_res_blocks": 1,
        "attn_resolutions": [],
        "dropout": 0.0,
    }
    unet = {
        "image_size": 8,
        "in_channels": 4,
        "model_channels": 32,
        "out_channels": 2,
        "num_res_blocks": 1,
        "attention_resolutions": [],
        "channel_mult": [1, 2],
        "num_head_channels": 16,
    }
    return {
        "apply_normalization": False,
        "encode_conditioning": True,
        "first_stage_config": first_stage,
        "cond_stage_config": unet,
        "denoiser_settings": {
            "timesteps": 10,
            "linear_start": 1e-4,
            "linear_end": 1e-2,
            "parameterization": "eps",
        },
        "other": {
            "concat_mode": True,
            "cond_stage_trainable": False,
            "first_stage_key": "image",
            "cond_stage_key": "LR_image",
        },
    }
