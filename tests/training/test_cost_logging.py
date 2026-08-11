from __future__ import annotations

from types import SimpleNamespace

import pytest

import opensr_model.training.callbacks as callback_helpers
from opensr_model.training.callbacks import NvidiaPowerCostLogger
from opensr_model.training.cli import _build_callbacks
from opensr_model.training.config import load_training_config


class WandbTestLogger:
    def __init__(self) -> None:
        self.records: list[tuple[dict[str, float], int | None]] = []

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.records.append((metrics, step))


def test_cost_logger_integrates_gpu_and_extrapolated_energy(monkeypatch) -> None:
    logger = WandbTestLogger()
    trainer = SimpleNamespace(loggers=[logger], global_step=0)
    callback = NvidiaPowerCostLogger(
        electricity_eur_per_kwh=0.30,
        overhead_watts=200,
        every_n_steps=20,
    )
    monkeypatch.setattr(callback, "_read_gpu_power_watts", lambda: [300.0, 300.0])
    times = iter((100.0, 3700.0))
    monkeypatch.setattr(callback_helpers.time, "monotonic", lambda: next(times))

    callback._sample_and_log(trainer)
    trainer.global_step = 20
    callback._sample_and_log(trainer)

    metrics, step = logger.records[-1]
    assert step == 20
    assert metrics["cost/actual/gpu_0_power_kw"] == pytest.approx(0.3)
    assert metrics["cost/actual/gpu_1_power_kw"] == pytest.approx(0.3)
    assert metrics["cost/actual/gpu_power_kw"] == pytest.approx(0.6)
    assert metrics["cost/actual/interval_energy_kwh"] == pytest.approx(0.6)
    assert metrics["cost/actual/energy_kwh"] == pytest.approx(0.6)
    assert metrics["cost/actual/price_eur"] == pytest.approx(0.18)
    assert metrics["cost/extrapolated/system_power_kw"] == pytest.approx(0.8)
    assert metrics["cost/extrapolated/interval_energy_kwh"] == pytest.approx(0.8)
    assert metrics["cost/extrapolated/energy_kwh"] == pytest.approx(0.8)
    assert metrics["cost/extrapolated/price_eur"] == pytest.approx(0.24)


def test_cost_logger_restores_totals_without_charging_process_downtime() -> None:
    callback = NvidiaPowerCostLogger()
    callback.load_state_dict({"gpu_energy_kwh": 4.0, "extrapolated_energy_kwh": 5.5})

    assert callback.gpu_energy_kwh == pytest.approx(4.0)
    assert callback.extrapolated_energy_kwh == pytest.approx(5.5)
    assert callback._previous_time is None
    assert callback._previous_gpu_power_watts is None


def test_wandb_configuration_adds_cost_callback(tmp_path) -> None:
    config = load_training_config(
        "opensr_model/configs/train_autoencoder.yaml",
        overrides=[
            "logging.type=wandb",
            "logging.cost.overhead_watts=225",
            "logging.images.enabled=false",
            "trainer.enable_checkpointing=false",
            "checkpoint.native.enabled=false",
        ],
        expected_stage="autoencoder",
    )

    callbacks = _build_callbacks(
        "autoencoder", config, tmp_path.resolve(), has_logger=True
    )
    cost = next(item for item in callbacks if isinstance(item, NvidiaPowerCostLogger))
    assert cost.electricity_eur_per_kwh == pytest.approx(0.30)
    assert cost.overhead_watts == pytest.approx(225)
    assert cost.every_n_steps == 20
