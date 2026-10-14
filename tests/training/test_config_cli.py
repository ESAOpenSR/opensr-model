"""Configuration failures and training-entrypoint orchestration contracts."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping

from opensr_model.training import cli
from opensr_model.training.runtime import config as configuration


@pytest.fixture
def config(tmp_path):
    result = configuration.load_training_config(
        configuration.package_config_path("train_autoencoder.yaml")
    )
    result.run_dir = str(tmp_path / "run")
    return result


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("stage", "bad", "stage must"),
        ("seed", -1, "seed"),
        ("model.architecture_config", "", "architecture_config"),
        ("data.taco_path", [], "taco_path"),
        ("data.taco_path", None, "taco_path"),
        ("data.train", "legacy", "Legacy data sources"),
        ("data.lr_asset_id", " ", "asset identifiers"),
        ("data.rgb_bands", [1, 1, 3], "distinct"),
        ("data.rgb_bands", [0, 2, 3], "positive indices"),
        ("data.nir_band", 0, "positive indices"),
        ("data.reflectance_scale", 0, "reflectance_scale"),
        ("data.hr_nir_strategy", "interpolate", "synthetic_sidecar"),
        ("data.val_fraction", 0, "strictly between"),
        ("data.val_fraction", 1, "strictly between"),
        ("data.factor", 2, "data.factor"),
        ("data.train_patch_size", 3, "train_patch_size"),
        ("data.validation_patch_size", -4, "validation_patch_size"),
        ("data.value_range", [1, 0], "increasing"),
        ("data.value_range", [0], "increasing"),
    ],
)
def test_training_config_rejects_invalid_contract(config, field, value, message):
    OmegaConf.update(config, field, value, merge=False)
    with pytest.raises(ValueError, match=message):
        configuration.validate_training_config(config)


@pytest.mark.parametrize(
    "section", ["model", "data", "training", "trainer", "checkpoint", "logging"]
)
def test_training_config_requires_each_section(config, section):
    del config[section]
    with pytest.raises(ValueError, match=section):
        configuration.validate_training_config(config)


def test_training_config_path_resolution_and_roundtrip(config, tmp_path, monkeypatch):
    architecture = tmp_path / "architecture.yaml"
    architecture.write_text(
        "first_stage_config: {}\ncond_stage_config: {}\ndenoiser_settings: {}\nother: {}\n"
    )
    config.model.architecture_config = "architecture.yaml"
    config.model.pretrained_checkpoint = "weights.ckpt"
    config.data.taco_path = ["one.taco", str(tmp_path / "two.taco")]
    config.run_dir = "results"
    path = tmp_path / "custom.yaml"
    OmegaConf.save(config, path)
    loaded = configuration.load_training_config(path, overrides=["seed=101"])
    assert loaded.seed == 101
    assert loaded.run_dir == str(tmp_path / "results")
    assert loaded.model.architecture_config == str(architecture)
    assert loaded.model.pretrained_checkpoint == str(tmp_path / "weights.ckpt")
    assert list(loaded.data.taco_path) == [
        str(tmp_path / "one.taco"),
        str(tmp_path / "two.taco"),
    ]
    assert configuration.load_model_config(architecture).other == {}
    with pytest.raises(ValueError, match="entrypoint"):
        configuration.load_training_config(path, expected_stage="diffusion")
    with pytest.raises(FileNotFoundError):
        configuration.load_training_config(tmp_path / "missing.yaml")
    architecture.write_text("other: {}\n")
    with pytest.raises(ValueError, match="first_stage_config"):
        configuration.load_model_config(architecture)
    plain = {"field": 3}
    assert configuration.load_model_config(plain).field == 3
    wrapped = OmegaConf.create(plain)
    assert configuration.load_model_config(wrapped) is wrapped
    assert configuration.as_plain_dict(plain) == plain
    with pytest.raises(TypeError, match="mapping"):
        configuration.as_plain_dict(OmegaConf.create([1, 2]))


@pytest.mark.parametrize(
    "kind", ["none", "false", "disabled", "csv", "tensorboard", "invalid"]
)
def test_logger_selection_and_paths(tmp_path, kind):
    options = OmegaConf.create(
        {"type": kind, "save_dir": "logs", "name": "run", "version": 7}
    )
    if kind == "invalid":
        with pytest.raises(ValueError, match="logging.type"):
            cli._build_logger(options, tmp_path)
    else:
        logger = cli._build_logger(options, tmp_path)
        if kind in {"none", "false", "disabled"}:
            assert logger is False
        else:
            assert Path(logger.log_dir) == tmp_path / "logs" / "run" / "7"


def test_wandb_options_are_forwarded_without_starting_network_session(
    tmp_path, monkeypatch
):
    logger = Mock()
    monkeypatch.setattr("pytorch_lightning.loggers.WandbLogger", logger)
    cli._build_logger(
        OmegaConf.create(
            {
                "type": "wandb",
                "project": "test",
                "name": "experiment",
                "entity": "team",
                "tags": ["smoke"],
                "mode": "offline",
                "save_dir": str(tmp_path),
            }
        ),
        tmp_path,
    )
    assert logger.call_args.kwargs["project"] == "test"
    assert logger.call_args.kwargs["mode"] == "offline"
    assert "resume" not in logger.call_args.kwargs


@pytest.mark.parametrize(
    "devices,accelerator,world,expected",
    [
        (1, "cpu", "1", False),
        (2, "cpu", "1", True),
        ([0, 1], "gpu", "1", True),
        ([0], "gpu", "1", False),
        ("auto", "cuda", "bad", True),
        ("auto", "cpu", "1", False),
        ("0,1", "gpu", "1", False),
        (1, "cpu", "2", True),
    ],
)
def test_multi_process_detection(monkeypatch, devices, accelerator, world, expected):
    monkeypatch.setenv("WORLD_SIZE", world)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert (
        cli._uses_multiple_processes({"devices": devices, "accelerator": accelerator})
        is expected
    )


@pytest.mark.parametrize(
    "options,message",
    [
        ({"accumulate_grad_batches": 2}, "accumulation"),
        ({"max_steps": 10}, "ambiguous"),
        ({"devices": 2, "strategy": "ddp"}, "find_unused_parameters"),
    ],
)
def test_manual_gan_trainer_rejects_incompatible_settings(
    monkeypatch, options, message
):
    monkeypatch.setenv("WORLD_SIZE", "1")
    with pytest.raises(ValueError, match=message):
        cli._prepare_trainer_kwargs(
            "autoencoder", SimpleNamespace(automatic_optimization=False), options
        )


def test_trainer_gradient_clipping_and_explicit_ddp_strategy(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "1")
    automatic = SimpleNamespace(automatic_optimization=True, gradient_clip_val=2.0)
    options = {}
    cli._prepare_trainer_kwargs("autoencoder", automatic, options)
    assert options["gradient_clip_val"] == 2
    options["gradient_clip_val"] = 3
    cli._prepare_trainer_kwargs("autoencoder", automatic, options)
    assert options["gradient_clip_val"] == 3
    options = {"devices": 2, "strategy": "ddp_find_unused_parameters_true"}
    cli._prepare_trainer_kwargs(
        "autoencoder", SimpleNamespace(automatic_optimization=False), options
    )
    assert options["strategy"] == "ddp_find_unused_parameters_true"
    cli._prepare_trainer_kwargs(
        "autoencoder", SimpleNamespace(automatic_optimization=False), {"devices": 1}
    )


@pytest.mark.parametrize(
    "entry,argv,stage",
    [
        (cli.main, ["autoencoder"], "autoencoder"),
        (cli.autoencoder_main, [], "autoencoder"),
        (cli.diffusion_main, [], "diffusion"),
    ],
)
def test_cli_print_config_does_not_start_training(
    entry, argv, stage, monkeypatch, capsys
):
    run = Mock()
    monkeypatch.setattr(cli, "run_training", run)
    entry([*argv, "--print-config", "--set", "seed=17"])
    printed = OmegaConf.create(capsys.readouterr().out)
    assert printed.stage == stage and printed.seed == 17
    run.assert_not_called()


def test_cli_checkpoint_overrides_and_mutually_exclusive_modes(tmp_path, monkeypatch):
    run = Mock()
    monkeypatch.setattr(cli, "run_training", run)
    cli.main(
        [
            "diffusion",
            "--resume",
            str(tmp_path / "resume.ckpt"),
            "--pretrained",
            str(tmp_path / "init.ckpt"),
            "--validate-only",
        ]
    )
    args, kwargs = run.call_args
    assert args[0].resume_checkpoint == str(tmp_path / "resume.ckpt")
    assert args[0].model.pretrained_checkpoint == str(tmp_path / "init.ckpt")
    assert kwargs == {"validate_only": True, "prepare_only": False}
    with pytest.raises(ValueError, match="mutually exclusive"):
        cli.main(["autoencoder", "--prepare-only", "--validate-only"])


@pytest.mark.parametrize("validate_only", [False, True])
@pytest.mark.parametrize("rank", ["0", "1"])
def test_run_training_freezes_data_paths_and_dispatches(
    config, tiny_model_config, tmp_path, monkeypatch, validate_only, rank
):
    config.model.architecture_config = tiny_model_config
    config.logging.type = "none"
    module = SimpleNamespace(automatic_optimization=True, gradient_clip_val=0)
    data = SimpleNamespace(
        taco_paths=[tmp_path / "part1.taco", tmp_path / "part2.taco"], factor=4
    )
    trainer = Mock()
    constructor = Mock(return_value=trainer)
    monkeypatch.setattr(cli, "OpenSRDataModule", Mock(return_value=data))
    monkeypatch.setattr(cli, "_build_module", Mock(return_value=module))
    monkeypatch.setattr(cli.pl, "Trainer", constructor)
    monkeypatch.setattr(cli.torch, "set_float32_matmul_precision", Mock())
    monkeypatch.setenv("LOCAL_RANK", rank)
    cli.run_training(config, validate_only=validate_only)
    method = trainer.validate if validate_only else trainer.fit
    method.assert_called_once_with(module, datamodule=data, ckpt_path=None)
    assert constructor.call_args.kwargs["default_root_dir"] == config.run_dir
    saved = Path(config.run_dir) / "resolved_config.yaml"
    assert saved.exists() == (rank == "0")
    if saved.exists():
        assert list(OmegaConf.load(saved).data.taco_path) == [
            str(p) for p in data.taco_paths
        ]


def test_prepare_only_returns_before_model_construction(config, monkeypatch):
    config.stage = "diffusion"
    config.training.validation = {"sen2neon": {}}
    prepare = Mock(return_value={"sen2neon_root": "/data/heldout"})
    monkeypatch.setattr(
        "opensr_model.training.validation.data.prepare_transfer", prepare
    )
    monkeypatch.setattr(cli.torch, "set_float32_matmul_precision", Mock())
    build = Mock()
    monkeypatch.setattr(cli, "_build_module", build)
    cli.run_training(config, prepare_only=True)
    assert config.checkpoint.monitor == "val/sen2neon/rgb_lpips"
    assert config.checkpoint.mode == "min"
    assert config.data.preparation_manifest.endswith("data_manifest.json")
    build.assert_not_called()
    config.trainer.enable_checkpointing = False
    with pytest.raises(ValueError, match="enable_checkpointing"):
        cli.run_training(config, prepare_only=True)
    config.stage = "autoencoder"
    with pytest.raises(ValueError, match="transfer-validation preset"):
        cli.run_training(config, prepare_only=True)


def test_module_construction_rejects_missing_initializer_and_scheduler(
    config, tiny_model_config
):
    config.model.pretrained_checkpoint = None
    config.model.autoencoder_checkpoint = None
    with pytest.raises(ValueError, match="requires model.pretrained_checkpoint"):
        cli._build_module("diffusion", OmegaConf.create(tiny_model_config), config)
    config.model.allow_random_first_stage = True
    config.training.scheduler.name = "bad"
    with pytest.raises(ValueError, match="scheduler"):
        cli._build_module("diffusion", OmegaConf.create(tiny_model_config), config)


def test_checkpoint_classification_and_early_stopping(config, tmp_path):
    path = tmp_path / "checkpoint.ckpt"
    assert not cli._checkpoint_has_full_diffusion_state(path)
    state = {
        name: torch.ones(1)
        for name in (
            "betas",
            "model.diffusion_model.weight",
            "model_ema.weight",
            "first_stage_model.encoder.weight",
            "first_stage_model.decoder.weight",
        )
    }
    torch.save({"state_dict": state, "optimizer_states": [{}], "loops": {}}, path)
    assert cli._checkpoint_has_full_diffusion_state(path)
    cli._validate_resume_checkpoint(path)
    config.model.pretrained_checkpoint = str(path)
    config.checkpoint.native.autoencoder_base_checkpoint = "auto"
    config.checkpoint.early_stopping = {"enabled": True, "patience": 3}
    callbacks = cli._build_callbacks("autoencoder", config, tmp_path, has_logger=True)
    assert any(
        isinstance(callback, EarlyStopping) and callback.patience == 3
        for callback in callbacks
    )
    native = next(c for c in callbacks if isinstance(c, cli.NativeCheckpointCallback))
    assert str(native.autoencoder_base_checkpoint) == str(path)
