"""Checkpoint failures must preserve the previous weights and serialized state."""

from collections import OrderedDict
from copy import deepcopy

import pytest
import torch

from opensr_model.training.runtime import checkpoints as ckpt


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("container", [None, "model_state_dict", "weights", "module"])
def test_load_supported_serializers_and_containers(tmp_path, legacy, container):
    state = torch.nn.Linear(3, 2).state_dict()
    payload = state if container is None else {container: state, "epoch": 7}
    path = tmp_path / "weights.pt"
    torch.save(payload, path, _use_new_zipfile_serialization=not legacy)
    loaded = ckpt.load_checkpoint_payload(path)
    assert list(loaded["state_dict"]) == list(state)
    assert loaded["state_dict"]._metadata == state._metadata
    for key, tensor in state.items():
        torch.testing.assert_close(loaded["state_dict"][key], tensor)
    if container is not None:
        assert loaded["epoch"] == 7


@pytest.mark.parametrize("payload", [42, [], {"weights": "bad"}, {"state_dict": {}}])
def test_bad_checkpoint_payloads_fail_with_format_error(tmp_path, payload):
    path = tmp_path / "bad.pt"
    torch.save(payload, path)
    with pytest.raises(ckpt.CheckpointFormatError):
        ckpt.load_checkpoint_payload(path)


def test_corrupt_and_missing_checkpoint_files(tmp_path):
    path = tmp_path / "bad.pt"
    with pytest.raises(ckpt.CheckpointFormatError, match="not a file"):
        ckpt.load_checkpoint_payload(path)
    path.write_bytes(b"incomplete checkpoint")
    with pytest.raises(ckpt.CheckpointFormatError, match="safely load"):
        ckpt.load_checkpoint_payload(path)


@pytest.mark.parametrize("failure", ["write", "replace"])
def test_failed_save_preserves_previous_checkpoint(tmp_path, monkeypatch, failure):
    path = tmp_path / "last.pt"
    ckpt.save_native_checkpoint(path, {"weight": torch.ones(2)}, epoch=1, global_step=2)
    previous_bytes = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("simulated full disk")

    with monkeypatch.context() as patch:
        patch.setattr(
            ckpt.torch if failure == "write" else ckpt.os,
            "save" if failure == "write" else "replace",
            fail,
        )
        with pytest.raises(OSError, match="full disk"):
            ckpt.save_native_checkpoint(
                path, {"weight": torch.zeros(2)}, epoch=2, global_step=4
            )
    assert path.read_bytes() == previous_bytes
    assert list(tmp_path.iterdir()) == [path]
    assert ckpt.load_checkpoint_payload(path)["epoch"] == 1


def test_save_on_filesystem_without_fsync(tmp_path, monkeypatch):
    def unsupported(*args):
        raise OSError("unsupported")

    monkeypatch.setattr(ckpt.os, "fsync", unsupported)
    path = tmp_path / "weights.pt"
    ckpt.save_native_checkpoint(path, {"weight": torch.ones(2)}, epoch=0, global_step=0)
    torch.testing.assert_close(
        ckpt.load_checkpoint_payload(path)["state_dict"]["weight"], torch.ones(2)
    )


def test_metadata_round_trip_remains_weights_only_safe(tmp_path):
    parameter = torch.nn.Parameter(torch.tensor([3.0]))
    metadata = {
        "path": tmp_path,
        "nested": [{"tensor": parameter}, (None, True, b"rgbn", 3, 0.5)],
    }
    path = tmp_path / "weights.pt"
    ckpt.save_native_checkpoint(
        path, {"weight": parameter}, epoch=2, global_step=3, metadata=metadata
    )
    loaded = torch.load(path, weights_only=True)
    assert loaded["metadata"]["path"] == str(tmp_path)
    assert loaded["metadata"]["nested"][1] == (None, True, b"rgbn", 3, 0.5)
    assert not loaded["metadata"]["nested"][0]["tensor"].requires_grad
    assert metadata["path"] == tmp_path
    assert parameter.requires_grad


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"epoch": True}, TypeError),
        ({"epoch": 1.5}, TypeError),
        ({"global_step": False}, TypeError),
        ({"epoch": -1}, ValueError),
        ({"global_step": -1}, ValueError),
        ({"metadata": []}, TypeError),
        ({"metadata": {1: "bad"}}, ckpt.CheckpointFormatError),
        ({"metadata": {"nested": [object()]}}, ckpt.CheckpointFormatError),
        ({"state_dict": {}}, ckpt.CheckpointFormatError),
        ({"state_dict": {1: torch.ones(1)}}, ckpt.CheckpointFormatError),
        ({"state_dict": {"weight": "bad"}}, ckpt.CheckpointFormatError),
    ],
)
def test_invalid_save_arguments_never_create_files(tmp_path, overrides, error):
    args = {"state_dict": {"weight": torch.ones(1)}, "epoch": 0, "global_step": 0}
    args.update(overrides)
    with pytest.raises(error):
        ckpt.save_native_checkpoint(tmp_path / "weights.pt", **args)
    assert not list(tmp_path.iterdir())


def test_legacy_loss_filter_matches_module_segments_and_preserves_metadata():
    state = OrderedDict(
        (key, torch.ones(1))
        for key in ("model.loss.weight", "loss.bias", "lossless.weight", "model.weight")
    )
    state._metadata = {"": {"version": 1}}
    kept, removed = ckpt.filter_legacy_loss_keys(state)
    assert list(kept) == ["lossless.weight", "model.weight"]
    assert removed == ("loss.bias", "model.loss.weight")
    assert kept._metadata == state._metadata
    assert len(state) == 4


def test_autoencoder_extraction_prefers_native_first_stage():
    roots = ("encoder", "decoder", "quant_conv", "post_quant_conv")
    state = {f"teacher.{root}.weight": torch.zeros(2) for root in roots}
    state.update(
        {f"module.first_stage_model.{root}.weight": torch.ones(2) for root in roots}
    )
    state["module.first_stage_model.loss.weight"] = torch.zeros(1)
    extracted = ckpt.extract_autoencoder_state_dict(state)
    assert set(extracted) == {f"{root}.weight" for root in roots}
    assert all(torch.equal(value, torch.ones(2)) for value in extracted.values())
    del state["teacher.encoder.weight"]
    del state["module.first_stage_model.encoder.weight"]
    with pytest.raises(
        ckpt.CheckpointCompatibilityError, match="complete AutoencoderKL"
    ):
        ckpt.extract_autoencoder_state_dict(state)


@pytest.mark.parametrize(
    "defect,match",
    [
        ("no_model", "incomplete"),
        ("no_shadow", "incomplete"),
        ("no_decay", "incomplete"),
        ("collision", "collide"),
        ("shape", "shape"),
        ("orphan", "exactly cover"),
    ],
)
def test_ema_export_rejects_inconsistent_weights_without_mutation(defect, match):
    state = OrderedDict(
        {
            "model.a.b": torch.ones(2),
            "model_ema.ab": torch.zeros(2),
            "model_ema.decay": torch.tensor(0.9),
            "model_ema.num_updates": torch.tensor(4),
        }
    )
    if defect == "no_model":
        del state["model.a.b"]
    elif defect == "no_shadow":
        del state["model_ema.ab"]
    elif defect == "no_decay":
        del state["model_ema.decay"]
    elif defect == "collision":
        state["model.ab"] = torch.ones(2)
    elif defect == "shape":
        state["model_ema.ab"] = torch.zeros(3)
    else:
        state["model_ema.orphan"] = torch.zeros(2)
    before = deepcopy(state)
    with pytest.raises(ckpt.CheckpointCompatibilityError, match=match):
        ckpt.materialize_diffusion_ema_weights(state)
    assert state.keys() == before.keys()
    for key in state:
        torch.testing.assert_close(state[key], before[key])


def test_partial_weight_load_reports_missing_and_shape_mismatches():
    model = torch.nn.Linear(3, 2)
    previous_weight = model.weight.detach().clone()
    report = ckpt.load_autoencoder_weights(
        model,
        {"weight": torch.ones(5, 3), "bias": torch.full((2,), 0.25)},
        strict=False,
    )
    assert not report.is_exact
    assert report.matched_count == 1
    assert report.matched_keys == ("bias",)
    assert report.shape_mismatches[0].target_key == "weight"
    torch.testing.assert_close(model.weight, previous_weight)
    torch.testing.assert_close(model.bias, torch.full((2,), 0.25))


@pytest.mark.parametrize(
    "section,value,error",
    [
        ("first_stage_config", None, TypeError),
        ("cond_stage_config", [], TypeError),
        ("denoiser_settings", [], TypeError),
        ("other", [], TypeError),
        ("first_stage_config", {}, ValueError),
        ("denoiser_settings", {"timesteps": 0}, ValueError),
    ],
)
def test_state_spec_rejects_invalid_architecture(
    tiny_model_config, section, value, error
):
    config = deepcopy(tiny_model_config)
    config[section] = value
    with pytest.raises(error):
        ckpt.build_native_diffusion_state_spec(config)
