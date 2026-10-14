"""Exercise the ablation's real file and rendering pipeline without model downloads."""

import argparse
import json

import numpy as np
import pytest
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_origin

from opensr_model.ablations import temperature_eta as ablation


@pytest.mark.parametrize("text,expected", [("0, .5, 1", (0, 0.5, 1)), ("2,,", (2,))])
def test_parse_values(text, expected):
    assert ablation._parse_values(text) == expected


@pytest.mark.parametrize("text", ["", " , ", "-1,2"])
def test_parse_values_rejects_empty_or_negative_grid(text):
    with pytest.raises(argparse.ArgumentTypeError):
        ablation._parse_values(text)


@pytest.mark.parametrize("key", [None, "LR_image", "image", "tensor"])
@pytest.mark.parametrize("batched", [False, True])
def test_tensor_input_normalizes_reflectance_and_discards_extra_bands(
    tmp_path, key, batched
):
    value = torch.full((5, 4, 6), 5000, dtype=torch.int16)
    value[0, 0, 0] = -10
    value = value.unsqueeze(0) if batched else value
    path = tmp_path / "image.pt"
    torch.save(value if key is None else {key: value}, path)
    result = ablation._read_input(path)
    assert result.shape == (1, 4, 4, 6)
    assert result.dtype == torch.float32
    assert result[0, 0, 0, 0] == 0
    assert result[0, 1, 0, 0] == 0.5


@pytest.mark.parametrize(
    "value,error",
    [
        ({"unknown": 1}, TypeError),
        (torch.zeros(2, 4, 3, 3), ValueError),
        (torch.zeros(3, 3, 3), ValueError),
        (torch.zeros(3), ValueError),
    ],
)
def test_invalid_tensor_input(tmp_path, value, error):
    path = tmp_path / "bad.pth"
    torch.save(value, path)
    with pytest.raises(error):
        ablation._read_input(path)


@pytest.mark.parametrize("bands", [3, 4, 5])
def test_geotiff_input_reads_rgb_nir_in_order(tmp_path, bands):
    path = tmp_path / "input.tif"
    values = (
        np.arange(1, bands + 1, dtype="uint16")[:, None, None]
        * np.ones((bands, 4, 4), dtype="uint16")
        * 1000
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=bands,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(10, 50, 0.01, 0.01),
    ) as dst:
        dst.write(values)
    if bands < 4:
        with pytest.raises(ValueError, match="at least 4 bands"):
            ablation._read_input(path)
    else:
        torch.testing.assert_close(
            ablation._read_input(path)[0, :, 0, 0], torch.tensor([0.1, 0.2, 0.3, 0.4])
        )


def test_texture_selection_avoids_nodata_and_finds_edges():
    image = torch.ones(1, 4, 16, 32)
    image[..., :16] = 0
    image[..., 16::2] = 0.2
    selected, top, left = ablation._texture_crop(image, 8)
    assert top == 0 and left >= 16
    assert (selected > 0).all() and selected.var() > 0
    original = torch.ones(1, 4, 8, 8)
    result, top, left = ablation._texture_crop(original, 8)
    assert result is original and (top, left) == (0, 0)
    with pytest.raises(ValueError, match="at least"):
        ablation._texture_crop(original, 9)


def test_rgb_stretch_and_center_crop_handle_degenerate_inputs():
    assert ablation._rgb_bounds([torch.zeros(1, 4, 2, 2)]) == (0, 1)
    assert ablation._rgb_bounds([torch.full((1, 4, 2, 2), float("nan"))]) == (0, 1)
    low, high = ablation._rgb_bounds([torch.ones(1, 4, 820, 820)])
    assert low == 1 and high > low
    image = torch.arange(64).reshape(1, 1, 8, 8).expand(1, 4, -1, -1).float()
    torch.testing.assert_close(ablation._detail_crop(image), image[..., 2:6, 2:6])
    rgb = ablation._rgb(image, (10, 20))
    assert rgb.shape == (8, 8, 3) and rgb.min() == 0 and rgb.max() == 1


def test_render_writes_inspectable_images_and_reproducible_grid_metadata(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.pt"
    config = tmp_path / "model.yaml"
    checkpoint = tmp_path / "weights.ckpt"
    torch.save(torch.full((1, 4, 8, 8), 0.3), source)
    config.write_text("{}")
    calls = []

    class TinySampler:
        def __init__(self, config, device):
            assert device == "cpu"

        def load_pretrained(self, path):
            assert path == str(checkpoint)

        def __call__(self, image, **kwargs):
            noise = torch.rand(1).item()
            calls.append((kwargs, noise))
            return (
                image.repeat_interleave(4, -1).repeat_interleave(4, -2) + noise * 0.01
            )

    monkeypatch.setattr(ablation, "SRLatentDiffusion", TinySampler)
    result = ablation.render_ablation(
        input_path=source,
        checkpoint=checkpoint,
        config_path=config,
        output_dir=tmp_path / "output",
        etas=[0, 0.95],
        temperatures=[0, 1],
        steps=3,
        seed=11,
        crop_size=8,
        device="cpu",
    )
    assert len(calls) == 4
    assert len({noise for _, noise in calls}) == 1
    assert [(kw["sampling_eta"], kw["sampling_temperature"]) for kw, _ in calls] == [
        (0, 0),
        (0, 1),
        (0.95, 0),
        (0.95, 1),
    ]
    for path in (result, result.parent / "selected_source_crop.png"):
        with Image.open(path) as image:
            assert image.width > 100 and image.height > 100
            image.verify()
    metadata = json.loads(result.with_suffix(".json").read_text())
    assert metadata["sampling_steps"] == 3 and metadata["seed"] == 11
    assert metadata["lr_crop"] == {"top": 0, "left": 0, "size": 8}
    assert metadata["rgb_stretch"]["high"] > metadata["rgb_stretch"]["low"]


def test_ablation_cli_passes_grid_options(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        ablation,
        "render_ablation",
        lambda **kw: calls.append(kw) or tmp_path / "result.png",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "ablation",
            "--input",
            "image.pt",
            "--checkpoint",
            "weights.ckpt",
            "--etas",
            "0,.5",
            "--temperatures",
            "1",
            "--steps",
            "7",
            "--device",
            "cpu",
        ],
    )
    ablation.main()
    assert calls[0]["etas"] == (0, 0.5) and calls[0]["steps"] == 7
    assert "result.png" in capsys.readouterr().out
