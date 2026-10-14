"""Training telemetry and previews should tolerate unavailable external services."""

import warnings
from types import SimpleNamespace
from unittest.mock import Mock

import matplotlib.pyplot as plt
import pytest
import torch

from opensr_model.training.runtime import callbacks
from opensr_model.training.runtime.checkpoints import load_checkpoint_payload


@pytest.mark.parametrize("output", ["", "N/A\n", "-1\n", "nan\n", "inf\n"])
def test_gpu_telemetry_rejects_missing_and_invalid_measurements(monkeypatch, output):
    monkeypatch.setattr(
        callbacks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    with pytest.raises(ValueError):
        callbacks.NvidiaPowerCostLogger()._read_gpu_power_watts()


def test_gpu_telemetry_recovers_without_charging_unobserved_time(monkeypatch):
    cb = callbacks.NvidiaPowerCostLogger(every_n_steps=2)
    records = []

    class WandbProbe:
        def log_metrics(self, metrics, step):
            records.append(metrics)

    trainer = SimpleNamespace(global_step=0, loggers=[WandbProbe()])
    outputs = iter(
        ["100\n200\n", "N/A", "N/A", "100\n200\n", "200\n200\n", "200\n200\n"]
    )
    monkeypatch.setattr(
        callbacks.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=next(outputs)),
    )
    times = iter([0.0, 7200.0, 10800.0, 10800.0])
    monkeypatch.setattr(callbacks.time, "monotonic", lambda: next(times))
    cb.on_fit_start(trainer, None)
    with pytest.warns(RuntimeWarning, match="telemetry recovers"):
        cb.on_validation_start(trainer, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cb.on_validation_end(trainer, None)
    assert not caught
    trainer.global_step = 2
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert cb.gpu_energy_kwh == 0
    cb.on_train_batch_end(trainer, None, None, None, 0)
    assert len(records) == 2
    trainer.global_step = 4
    cb.on_train_batch_end(trainer, None, None, None, 0)
    cb.on_fit_end(trainer, None)
    assert cb.gpu_energy_kwh == pytest.approx(0.35)
    assert cb.state_dict()["extrapolated_energy_kwh"] == pytest.approx(0.55)


@pytest.mark.parametrize(
    "kwargs",
    [{"electricity_eur_per_kwh": -1}, {"overhead_watts": -1}, {"every_n_steps": 0}],
)
def test_cost_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        callbacks.NvidiaPowerCostLogger(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_images": 0},
        {"every_n_epochs": 0},
        {"value_max": 0},
        {"stretch_percentiles": (99, 1)},
    ],
)
def test_preview_options_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):
        callbacks.ReconstructionImageLogger(**kwargs)


@pytest.mark.parametrize(
    "outputs,error",
    [
        ({"images": {"hr": torch.ones(4, 8, 8)}}, ValueError),
        ({"images": {"hr": torch.ones(2, 4, 8, 8), "lr": None}}, ValueError),
        (
            {"images": {"hr": torch.ones(2, 4, 8, 8), "lr": torch.ones(1, 4, 8, 8)}},
            ValueError,
        ),
        ({"images": {"hr": torch.ones(2, 4, 8, 8)}, "details": []}, TypeError),
        (
            {"images": {"hr": torch.ones(2, 4, 8, 8)}, "details": {"hr": None}},
            ValueError,
        ),
        (
            {
                "images": {"hr": torch.ones(2, 4, 8, 8)},
                "details": {"hr": torch.ones(1, 4, 8, 8)},
            },
            ValueError,
        ),
    ],
)
def test_preview_panels_require_compatible_batch_dimensions(outputs, error):
    cb = callbacks.ReconstructionImageLogger()
    with pytest.raises(error):
        cb.on_validation_batch_end(
            SimpleNamespace(is_global_zero=True), None, outputs, None, 0
        )


def test_preview_caps_images_and_supplies_stable_default_ids():
    cb = callbacks.ReconstructionImageLogger(max_images=1)
    trainer = SimpleNamespace(is_global_zero=True)
    for output in (None, {}, {"images": None}, {"images": {}}):
        cb.on_validation_batch_end(trainer, None, output, None, 0)
    for batch_index in range(2):
        cb.on_validation_batch_end(
            trainer, None, {"images": {"hr": torch.ones(2, 4, 8, 8)}}, None, batch_index
        )
    assert cb._sample_ids == ["batch0000_00"]
    assert cb._images["hr"][0].shape[0] == 1


def test_preview_paths_and_degenerate_stretch_fallback(tmp_path):
    trainer = SimpleNamespace(default_root_dir=tmp_path, log_dir=None)
    cb = callbacks.ReconstructionImageLogger()
    assert cb._resolve_output_dir(trainer) == tmp_path / "images"
    trainer.log_dir = str(tmp_path / "logs")
    assert cb._resolve_output_dir(trainer) == tmp_path / "logs/images"
    assert (
        callbacks.ReconstructionImageLogger(output_dir="preview")._resolve_output_dir(
            trainer
        )
        == tmp_path / "preview"
    )
    for values in (
        [torch.full((1, 3, 4, 4), float("nan"))],
        [torch.ones(1, 1, 4, 4)],
        [torch.ones(1, 3, 4, 4)],
    ):
        assert cb._shared_rgb_stretch(values) == (0.0, 0.3)


def test_figures_forward_to_tensorboard_without_network():
    cb = callbacks.ReconstructionImageLogger()
    experiment = SimpleNamespace(add_figure=Mock())
    trainer = SimpleNamespace(
        loggers=[SimpleNamespace(experiment=experiment)], global_step=7
    )
    figure = cb._make_figure({"hr": torch.rand(1, 3, 4, 4)}, ["scene"], mode="rgb")
    try:
        cb._log_figures(trainer, {"images/rgb": figure})
        experiment.add_figure.assert_called_once_with(
            "images/rgb", figure, global_step=7
        )
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "mean"},
        {"every_n_epochs": 0},
        {"save_last": False, "save_best": False},
        {"weight_source": "invalid"},
    ],
)
def test_native_callback_rejects_unsafe_settings(tmp_path, kwargs):
    with pytest.raises(ValueError):
        callbacks.NativeCheckpointCallback(dirpath=str(tmp_path), **kwargs)


def test_best_only_export_updates_score_after_successful_save(tmp_path, monkeypatch):
    cb = callbacks.NativeCheckpointCallback(
        dirpath="native", save_last=False, mode="max"
    )
    trainer = SimpleNamespace(
        default_root_dir=tmp_path,
        current_epoch=0,
        global_step=1,
        sanity_checking=False,
        callback_metrics={"val/loss": torch.tensor(0.5)},
    )
    module = SimpleNamespace(
        checkpoint_stage="autoencoder",
        native_state_dict=lambda: {"weight": torch.ones(2)},
    )
    cb.on_validation_end(trainer, module)
    assert cb.best_score == 0.5
    assert not (tmp_path / "native/last-autoencoder.ckpt").exists()
    path = tmp_path / "native/best-autoencoder.ckpt"
    assert load_checkpoint_payload(path)["global_step"] == 1
    previous = path.read_bytes()
    trainer.callback_metrics["val/loss"] = torch.tensor(0.6)

    def fail(*args):
        raise OSError("disk unavailable")

    monkeypatch.setattr(cb, "_save", fail)
    with pytest.raises(OSError):
        cb.on_validation_end(trainer, module)
    assert cb.best_score == 0.5 and path.read_bytes() == previous
    cb.load_state_dict({"best_score": float("nan")})
    assert cb.best_score == 0.5
