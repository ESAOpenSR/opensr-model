from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import opensr_model.training.callbacks as callback_helpers
from opensr_model.autoencoder.autoencoder import AutoencoderKL
from opensr_model.diffusion.latentdiffusion import LatentDiffusion
from opensr_model.training.callbacks import NativeCheckpointCallback
from opensr_model.training.checkpoints import (
    CheckpointCompatibilityError,
    build_native_diffusion_state_spec,
    load_autoencoder_weights,
    load_checkpoint_payload,
    load_diffusion_weights,
    materialize_diffusion_ema_weights,
    merge_autoencoder_checkpoint,
    remap_state_dict,
    save_native_checkpoint,
)
from opensr_model.training.config import load_model_config


def _autoencoder(config: dict) -> AutoencoderKL:
    first_stage = config["first_stage_config"]
    return AutoencoderKL(first_stage, embed_dim=first_stage["embed_dim"])


def _diffusion(config: dict) -> LatentDiffusion:
    first = config["first_stage_config"]
    unet = config["cond_stage_config"]
    settings = config["denoiser_settings"]
    other = config["other"]
    return LatentDiffusion(
        first,
        unet,
        timesteps=settings["timesteps"],
        unet_config=unet,
        linear_start=settings["linear_start"],
        linear_end=settings["linear_end"],
        concat_mode=other["concat_mode"],
        cond_stage_trainable=other["cond_stage_trainable"],
        first_stage_key=other["first_stage_key"],
        cond_stage_key=other["cond_stage_key"],
        parameterization=settings["parameterization"],
    )


def test_autoencoder_loader_removes_lightning_wrapper_prefix(tiny_model_config) -> None:
    source = _autoencoder(tiny_model_config)
    wrapped = OrderedDict(
        (f"autoencoder.{key}", value.clone())
        for key, value in source.state_dict().items()
    )
    wrapped["discriminator.network.0.weight"] = torch.randn(1)
    target = _autoencoder(tiny_model_config)
    report = load_autoencoder_weights(target, {"state_dict": wrapped})
    assert report.is_exact
    assert report.source_prefix == "autoencoder."
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_native_diffusion_round_trip_and_autoencoder_merge(
    tmp_path, tiny_model_config
) -> None:
    base = _diffusion(tiny_model_config)
    base_path = tmp_path / "base.ckpt"
    save_native_checkpoint(base_path, base.state_dict(), epoch=0, global_step=0)

    trained_ae = OrderedDict(
        (key, value.clone())
        for key, value in base.first_stage_model.state_dict().items()
    )
    trained_ae["quant_conv.bias"].add_(0.25)
    merged_path = tmp_path / "merged.ckpt"
    merge_autoencoder_checkpoint(
        base_path,
        trained_ae,
        merged_path,
        expected_full_state_dict=base.state_dict(),
        epoch=3,
        global_step=17,
        metadata={"test": True},
    )

    payload = load_checkpoint_payload(merged_path)
    assert tuple(payload["state_dict"]) == tuple(base.state_dict())
    assert payload["metadata"]["test"] is True
    assert torch.equal(
        payload["state_dict"]["first_stage_model.quant_conv.bias"],
        trained_ae["quant_conv.bias"],
    )
    assert torch.equal(
        payload["state_dict"]["model.diffusion_model.time_embed.0.weight"],
        base.state_dict()["model.diffusion_model.time_embed.0.weight"],
    )

    restored = _diffusion(tiny_model_config)
    report = load_diffusion_weights(restored, merged_path)
    assert report.is_exact
    assert len(restored.state_dict()) == len(base.state_dict())


def test_materialized_ema_preserves_native_contract_and_loads_strictly(
    tmp_path, tiny_model_config
) -> None:
    model = _diffusion(tiny_model_config)
    original = OrderedDict(
        (key, tensor.detach().clone()) for key, tensor in model.state_dict().items()
    )
    raw_key = next(key for key in original if key.startswith("model.diffusion_model."))
    shadow_key = "model_ema." + raw_key.removeprefix("model.").replace(".", "")
    original[raw_key].zero_()
    original[shadow_key].fill_(0.375)

    materialized = materialize_diffusion_ema_weights(original)

    assert tuple(materialized) == tuple(original)
    assert torch.count_nonzero(original[raw_key]) == 0
    assert torch.equal(materialized[raw_key], original[shadow_key])
    assert torch.equal(materialized[shadow_key], original[shadow_key])
    build_native_diffusion_state_spec(tiny_model_config).validate(
        materialized, context="EMA-materialized state_dict"
    )

    path = tmp_path / "ema.ckpt"
    save_native_checkpoint(path, materialized, epoch=1, global_step=2)
    restored = _diffusion(tiny_model_config)
    report = load_diffusion_weights(restored, path)
    assert report.is_exact
    assert torch.equal(restored.state_dict()[raw_key], original[shadow_key])


def test_materialized_ema_rejects_incomplete_shadow_state(
    tiny_model_config,
) -> None:
    state = OrderedDict(_diffusion(tiny_model_config).state_dict())
    shadow_key = next(
        key
        for key in state
        if key.startswith("model_ema.")
        and key not in {"model_ema.decay", "model_ema.num_updates"}
    )
    state.pop(shadow_key)
    with pytest.raises(CheckpointCompatibilityError, match="exactly cover"):
        materialize_diffusion_ema_weights(state)


def test_meta_state_spec_exactly_matches_current_architecture(
    tiny_model_config,
) -> None:
    expected = _diffusion(tiny_model_config).state_dict()
    spec = build_native_diffusion_state_spec(tiny_model_config)
    assert spec.entries == tuple(
        (key, tuple(tensor.shape)) for key, tensor in expected.items()
    )


def test_merge_rejects_base_missing_an_architecture_tensor(
    tmp_path, tiny_model_config
) -> None:
    model = _diffusion(tiny_model_config)
    malformed = OrderedDict(model.state_dict())
    malformed.pop("model.diffusion_model.time_embed.0.weight")
    base_path = tmp_path / "malformed.ckpt"
    save_native_checkpoint(base_path, malformed, epoch=0, global_step=0)
    output_path = tmp_path / "must-not-exist.ckpt"

    with pytest.raises(CheckpointCompatibilityError, match="expected architecture"):
        merge_autoencoder_checkpoint(
            base_path,
            model.first_stage_model.state_dict(),
            output_path,
            expected_full_state_dict=build_native_diffusion_state_spec(
                tiny_model_config
            ),
            epoch=1,
            global_step=2,
        )
    assert not output_path.exists()


def test_prefix_remapping_rejects_duplicate_equally_exact_components() -> None:
    target = OrderedDict({"weight": torch.empty(2, 3)})
    source = OrderedDict(
        {
            "module.student.weight": torch.zeros(2, 3),
            "teacher.weight": torch.ones(2, 3),
        }
    )
    with pytest.raises(CheckpointCompatibilityError, match="ambiguous"):
        remap_state_dict(source, target)


def test_merge_rejects_duplicate_wrapped_full_models(
    tmp_path, tiny_model_config
) -> None:
    model = _diffusion(tiny_model_config)
    duplicated = OrderedDict()
    for key, tensor in model.state_dict().items():
        duplicated[f"teacher.{key}"] = tensor
        duplicated[f"module.student.{key}"] = tensor

    with pytest.raises(CheckpointCompatibilityError, match="ambiguous"):
        merge_autoencoder_checkpoint(
            {"state_dict": duplicated},
            model.first_stage_model.state_dict(),
            tmp_path / "must-not-exist.ckpt",
            expected_full_state_dict=build_native_diffusion_state_spec(
                tiny_model_config
            ),
            epoch=1,
            global_step=2,
        )


class _NativeTestModule(torch.nn.Module):
    checkpoint_stage = "diffusion"

    def __init__(self, value: float) -> None:
        super().__init__()
        self.register_buffer("value", torch.tensor(value))

    def native_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict({"value": self.value.detach().clone()})

    def opensr_inference_config(self) -> dict[str, object]:
        return {"denoiser_settings": {"sampling_steps": 100}}


class _NativeAutoencoderTestModule(torch.nn.Module):
    checkpoint_stage = "autoencoder"

    def __init__(self, model_config: dict, state: OrderedDict) -> None:
        super().__init__()
        self.model_config = model_config
        self._state = state

    def native_state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return self._state


def _run_native_validation_end(
    callback: NativeCheckpointCallback,
    trainer: SimpleNamespace,
    module: _NativeTestModule,
) -> None:
    hook = NativeCheckpointCallback.on_validation_end.__wrapped__
    hook(callback, trainer, module)


def test_native_callback_keeps_last_and_best_destinations_independent(
    tmp_path,
) -> None:
    callback = NativeCheckpointCallback(dirpath="native", weight_source="raw")
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        callback_metrics={"val/loss": torch.tensor(0.5)},
        default_root_dir=tmp_path,
        global_step=1,
    )
    first = _NativeTestModule(1.0)
    _run_native_validation_end(callback, trainer, first)

    trainer.current_epoch = 1
    trainer.global_step = 2
    trainer.callback_metrics["val/loss"] = torch.tensor(0.8)
    second = _NativeTestModule(2.0)
    _run_native_validation_end(callback, trainer, second)

    best = load_checkpoint_payload(tmp_path / "native" / "best-inference.ckpt")
    last = load_checkpoint_payload(tmp_path / "native" / "last-inference.ckpt")
    assert best["state_dict"]["value"].item() == pytest.approx(1.0)
    assert last["state_dict"]["value"].item() == pytest.approx(2.0)
    assert callback.best_score == pytest.approx(0.5)
    assert last["metadata"]["inference_unet_weight_source"] == "raw"

    restored = NativeCheckpointCallback(dirpath="native", weight_source="raw")
    assert restored.state_key == callback.state_key
    restored.load_state_dict(callback.state_dict())
    assert restored.best_score == pytest.approx(0.5)
    assert (
        NativeCheckpointCallback(dirpath="elsewhere", weight_source="raw").state_key
        != callback.state_key
    )
    assert (
        NativeCheckpointCallback(dirpath="native", weight_source="ema").state_key
        != callback.state_key
    )


def test_native_callback_validates_and_merges_autoencoder_full_export(
    tmp_path, tiny_model_config
) -> None:
    base = _diffusion(tiny_model_config)
    base_path = tmp_path / "base.ckpt"
    save_native_checkpoint(base_path, base.state_dict(), epoch=0, global_step=0)
    autoencoder_state = OrderedDict(
        (key, tensor.detach().clone())
        for key, tensor in base.first_stage_model.state_dict().items()
    )
    autoencoder_state["quant_conv.bias"].add_(0.125)
    module = _NativeAutoencoderTestModule(tiny_model_config, autoencoder_state)
    callback = NativeCheckpointCallback(
        dirpath="native",
        save_best=False,
        autoencoder_base_checkpoint=str(base_path),
    )
    trainer = SimpleNamespace(
        sanity_checking=False,
        current_epoch=0,
        callback_metrics={"val/loss": torch.tensor(0.25)},
        default_root_dir=tmp_path,
        global_step=3,
    )
    hook = NativeCheckpointCallback.on_validation_end.__wrapped__
    hook(callback, trainer, module)

    exported = load_checkpoint_payload(tmp_path / "native" / "last-inference.ckpt")[
        "state_dict"
    ]
    build_native_diffusion_state_spec(tiny_model_config).validate(
        exported, context="Callback export state_dict"
    )
    assert torch.equal(
        exported["first_stage_model.quant_conv.bias"],
        autoencoder_state["quant_conv.bias"],
    )


def test_secure_clone_falls_back_to_copy_and_tolerates_unsupported_fsync(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "source.ckpt"
    destination = tmp_path / "destination.ckpt"
    source.write_bytes(b"complete checkpoint")
    destination.write_bytes(b"old")

    def unavailable(*args, **kwargs):
        raise OSError("unsupported")

    monkeypatch.setattr(callback_helpers.os, "link", unavailable)
    monkeypatch.setattr(callback_helpers.os, "fsync", unavailable)
    callback_helpers._clone_checkpoint(source, destination)

    assert destination.read_bytes() == b"complete checkpoint"
    assert not tuple(tmp_path.glob(".destination.ckpt.*"))


def test_native_callback_rejects_sharded_state_strategy(tmp_path) -> None:
    class FSDPStrategy:
        pass

    callback = NativeCheckpointCallback(dirpath=str(tmp_path))
    trainer = SimpleNamespace(strategy=FSDPStrategy())
    with pytest.raises(RuntimeError, match="unsharded"):
        callback.setup(trainer, _NativeTestModule(1.0))


def test_released_checkpoint_has_expected_native_contract() -> None:
    checkpoint = Path("opensr-ldsrs2_v1_0_0.ckpt")
    if not checkpoint.exists():
        pytest.skip("released checkpoint is not available in this checkout")
    state = load_checkpoint_payload(checkpoint)["state_dict"]
    assert len(state) == 830
    assert sum(key.startswith("first_stage_model.") for key in state) == 204
    assert sum(key.startswith("model.diffusion_model.") for key in state) == 306
    assert sum(key.startswith("model_ema.") for key in state) == 308
    assert state["betas"][0].item() == pytest.approx(0.0015)
    assert state["betas"][-1].item() == pytest.approx(0.0155)
    model_config = load_model_config("opensr_model/configs/config_10m.yaml")
    build_native_diffusion_state_spec(model_config).validate(
        state, context="Released checkpoint state_dict"
    )
