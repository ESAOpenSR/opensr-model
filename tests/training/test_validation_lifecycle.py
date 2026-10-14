"""Validation must restore training state, including when assessment fails."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import rasterio
import torch
from pytorch_lightning.trainer.states import TrainerFn
from rasterio.transform import from_origin

from opensr_model.training.modules.diffusion import DiffusionTrainingModule
from opensr_model.training.runtime.checkpoints import save_native_checkpoint
from opensr_model.training.validation import callback as callbacks


@pytest.fixture
def prepared():
    return {
        "request_hash": "fixed",
        "sen2neon_root": "/unused",
        "sen2neon_rows": [{"id": "a", "site": "one", "landcover": "Forest"}],
        "selection_ids": ["a"],
        "raw_rows": [],
    }


@pytest.fixture
def trainer():
    return SimpleNamespace(
        global_rank=0,
        world_size=1,
        global_step=4,
        current_epoch=1,
        is_global_zero=True,
        sanity_checking=False,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        logger=None,
        strategy=SimpleNamespace(broadcast=lambda value, src: value, barrier=Mock()),
        callback_metrics={},
    )


@pytest.mark.parametrize(
    "options",
    [
        {"every_n_train_steps": 0},
        {"sen2neon": {"preview_count": -1}},
        {"sen2neon": {"spectral_tolerance": -1}},
        {"sen2neon": {"spectral_tolerance": float("nan")}},
    ],
)
def test_validation_rejects_invalid_options(prepared, tmp_path, options):
    with pytest.raises(ValueError):
        callbacks.TransferValidation(prepared, options, tmp_path)


def test_rank_zero_writer_errors_propagate_to_all_ranks(trainer):
    with pytest.raises(
        RuntimeError, match="Validation output failed: OSError: disk full"
    ):
        with callbacks.rank_zero_section(trainer) as writer:
            assert writer
            raise OSError("disk full")
    trainer.is_global_zero = False
    trainer.strategy.broadcast = lambda value, src: "remote write failure"
    with pytest.raises(RuntimeError, match="remote write failure"):
        with callbacks.rank_zero_section(trainer) as writer:
            assert not writer


def test_gather_rejects_duplicate_ids_and_sample_failures():
    with pytest.raises(RuntimeError, match="duplicate sample IDs"):
        callbacks.gather_records([{"id": "a"}, {"id": "a"}])
    with pytest.raises(RuntimeError, match="rank 0: failed"):
        callbacks.gather_records([], "rank 0: failed")
    assert callbacks.gather_records([{"id": "b"}, {"id": "a"}]) == [
        {"id": "a"},
        {"id": "b"},
    ]


def test_sample_uses_stable_per_scene_seed_and_rejects_nan(prepared, trainer, tmp_path):
    cb = callbacks.TransferValidation(
        prepared,
        {"seed": 9, "sampling_steps": 3, "sampling_eta": 0.2, "temperature": 0.7},
        tmp_path,
    )
    lr = torch.ones(1, 4, 4, 4)
    module = SimpleNamespace(
        device=torch.device("cpu"), sample_super_resolution=Mock(return_value=lr.half())
    )
    for scene in ["a", "b", "a"]:
        result = cb._sample(trainer, module, lr, scene)
        assert result.dtype == torch.float32
    calls = module.sample_super_resolution.call_args_list
    assert calls[0].kwargs == calls[2].kwargs
    assert calls[0].kwargs["seed"] != calls[1].kwargs["seed"]
    assert {k: v for k, v in calls[0].kwargs.items() if k != "seed"} == {
        "steps": 3,
        "eta": 0.2,
        "temperature": 0.7,
        "use_ema": True,
    }
    module.sample_super_resolution.return_value = lr * float("nan")
    with pytest.raises(ValueError, match="Nonfinite reconstruction for a"):
        cb._sample(trainer, module, lr, "a")


def test_validation_schedule_counts_optimizer_steps_and_skips_sanity(
    prepared, trainer, tmp_path, monkeypatch
):
    cb = callbacks.TransferValidation(prepared, {"every_n_train_steps": 2}, tmp_path)
    check = Mock(
        side_effect=lambda *args: setattr(
            cb, "last_evaluated_step", trainer.global_step
        )
    )
    monkeypatch.setattr(cb, "_check", check)
    for step in [0, 1, 2, 2, 3, 4, 4]:
        trainer.global_step = step
        cb.on_train_batch_end(trainer, None, None, None, 0)
    assert check.call_count == 2
    evaluate = Mock()
    monkeypatch.setattr(cb, "_evaluate", evaluate)
    trainer.sanity_checking = True
    cb.on_validation_epoch_end(trainer, None)
    trainer.sanity_checking = False
    cb.on_validation_epoch_end(trainer, None)
    evaluate.assert_not_called()
    trainer.state.fn = TrainerFn.VALIDATING
    cb.on_validation_epoch_end(trainer, None)
    evaluate.assert_called_once_with(trainer, None, cb.selection_rows, "validate_only")


def test_resuming_validation_requires_original_native_exports(
    prepared, trainer, tmp_path
):
    cb = callbacks.TransferValidation(prepared, {}, tmp_path)
    cb.baseline = {"rgb_lpips": 0.5}
    with pytest.raises(FileNotFoundError, match="original baseline"):
        cb.on_fit_start(trainer, None)
    path = tmp_path / "checkpoints/native/baseline-inference.ckpt"
    path.parent.mkdir(parents=True)
    path.touch()
    with pytest.raises(FileNotFoundError, match="original best"):
        cb.on_fit_start(trainer, None)


def test_missing_tile_metric_cannot_update_selection_or_write_checkpoint(
    prepared, trainer, tmp_path, monkeypatch
):
    cb = callbacks.TransferValidation(prepared, {}, tmp_path)
    monkeypatch.setattr(
        cb,
        "_evaluate",
        lambda *args: [
            {"id": "a", "metrics": {"rgb_lpips": 0.2, "nir_mae": None, "sam": 0.1}}
        ],
    )
    with pytest.raises(ValueError, match="no checkpoint selected"):
        cb._check(trainer, None, initial=True)
    assert cb.baseline is None and cb.last_evaluated_step == -1
    assert not (tmp_path / "checkpoints").exists()


@pytest.mark.parametrize("assessment_fails", [False, True])
def test_final_assessment_loads_each_export_and_restores_live_training_weights(
    prepared, trainer, tmp_path, monkeypatch, tiny_model_config, assessment_fails
):
    module = DiffusionTrainingModule(tiny_model_config, validation_num_samples=0)
    cb = callbacks.TransferValidation(prepared, {}, tmp_path)
    cb.last_evaluated_step = trainer.global_step
    cb.best_step = 2
    cb.baseline_evaluation, cb.best_evaluation, cb.last_evaluation = (
        "baseline-selection",
        "best-selection",
        "last-selection",
    )
    live = {
        key: value.detach().clone()
        for key, value in module.diffusion.state_dict().items()
    }
    key = "model.diffusion_model.time_embed.0.weight"
    for index, name in enumerate(("baseline", "best", "last")):
        state = dict(live)
        state[key] = torch.full_like(state[key], index + 1.0)
        save_native_checkpoint(
            tmp_path / "checkpoints/native" / f"{name}-inference.ckpt",
            state,
            epoch=0,
            global_step=index,
        )

    def save_resume(path):
        save_native_checkpoint(
            path,
            module.diffusion.state_dict(),
            epoch=1,
            global_step=trainer.global_step,
        )

    trainer.save_checkpoint = save_resume
    evaluated = []

    def evaluate(trainer, model, rows, label, reuse, save_all):
        name = label.rsplit("_", 1)[-1]
        expected = ("baseline", "best", "last").index(name) + 1.0
        torch.testing.assert_close(
            model.diffusion.state_dict()[key], torch.full_like(live[key], expected)
        )
        assert save_all and reuse == f"{name}-selection"
        assert rows == prepared["sen2neon_rows"]
        evaluated.append(name)
        if assessment_fails and name == "best":
            raise RuntimeError("assessment failed")
        return [
            {
                "id": "a",
                "site": "one",
                "metrics": {
                    "rgb_lpips": 0.5 if name == "baseline" else 0.4,
                    "nir_mae": 0.1,
                    "sam": 0.2,
                },
            }
        ]

    monkeypatch.setattr(cb, "_evaluate", evaluate)
    if assessment_fails:
        with pytest.raises(RuntimeError, match="assessment failed"):
            cb.on_train_end(trainer, module)
        assert evaluated == ["baseline", "best"]
        assert not (tmp_path / "validation/final_report.json").exists()
    else:
        cb.on_train_end(trainer, module)
        assert evaluated == ["baseline", "best", "last"]
        report = json.loads((tmp_path / "validation/final_report.json").read_text())
        assert report["best_selection_step"] == 2
        assert report["assessment_tiles"] == 1
        assert report["candidates"]["best"]["spectral_limits_pass"]
        assert report["candidates"]["last"]["paired_changes"]["metrics"]["rgb_lpips"][
            "mean_delta"
        ] == pytest.approx(-0.1)
    for name, value in module.diffusion.state_dict().items():
        torch.testing.assert_close(value, live[name])
    assert trainer.strategy.barrier.call_count >= 2


def test_raw_previews_preserve_extent_and_restore_rng_and_training_mode(
    prepared, trainer, tmp_path, monkeypatch, tiny_model_config
):
    raw = np.full((4, 8, 12), 2000, dtype=np.uint16)
    source = tmp_path / "raw.tif"
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        count=4,
        height=8,
        width=12,
        dtype="uint16",
        transform=from_origin(100, 200, 10, 10),
    ) as dst:
        dst.write(raw)
    prepared.update(
        sen2neon_root=None,
        sen2neon_rows=[],
        selection_ids=[],
        raw_rows=[
            {"id": "scene", "path": str(source), "bands": [1, 2, 3, 4], "scale": 1e-4}
        ],
    )
    cb = callbacks.TransferValidation(prepared, {"every_n_train_steps": 2}, tmp_path)
    module = DiffusionTrainingModule(tiny_model_config, validation_num_samples=0)
    module.train()

    def sample(lr, **kwargs):
        assert not module.training and torch.is_inference_mode_enabled()
        torch.rand(1)
        return torch.nn.functional.interpolate(lr, scale_factor=4, mode="nearest")

    monkeypatch.setattr(module, "sample_super_resolution", sample)
    before = torch.random.get_rng_state().clone()
    cb.on_fit_start(trainer, module)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert module.training
    output = tmp_path / "validation/raw_s2_step00000000"
    prediction = np.load(output / "scene.npz")["sr"]
    assert prediction.shape == (4, 32, 48)
    np.testing.assert_allclose(prediction, 0.2, atol=1e-4)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["outputs"] == [{"id": "scene", "file": "scene.npz"}]
    assert len(list((output / "previews").glob("*.png"))) == 12
    cb.on_train_batch_end(trainer, module, None, None, 0)
    assert cb.last_evaluated_step == 4
    trainer.state.fn = TrainerFn.VALIDATING
    cb.on_validation_epoch_end(trainer, module)
    assert (tmp_path / "validation/raw_s2_validate_only/manifest.json").exists()
