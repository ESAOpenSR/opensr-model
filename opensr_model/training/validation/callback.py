"""SEN2NEON validation, spectral-guarded selection, and raw S2 previews."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributed as dist
from pytorch_lightning.trainer.states import TrainerFn

from opensr_model.autoencoder.utils import AttnBlock
from ..runtime.checkpoints import (
    load_diffusion_weights,
    materialize_diffusion_ema_weights,
    save_native_checkpoint,
)
from .images import log_outputs, preview_row, safe_id, wandb_run
from .metrics import (
    aggregate,
    finite_json,
    guardrail_decision,
    paired_report,
    reconstruction_scores,
)
from .data import SEN2NEONDataset, pad_lr, read_raw, select_diverse, write_json


@contextlib.contextmanager
def validation_inference(module, query_chunk_size=512):
    """Do not alter training RNG, mode, parameters, or checkpoint structure."""
    blocks = [m for m in module.first_stage_model.modules() if isinstance(m, AttnBlock)]
    previous = [m.query_chunk_size for m in blocks]
    training = module.training
    devices = [module.device.index] if module.device.type == "cuda" else []
    try:
        for block in blocks:
            block.query_chunk_size = query_chunk_size
        module.eval()
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            yield
    finally:
        for block, size in zip(blocks, previous):
            block.query_chunk_size = size
        module.train(training)


def gather_records(local: list[dict], error: str | None = None) -> list[dict]:
    """One collective after unequal local shards; no per-sample collectives."""
    payload = {"records": local, "error": error}
    parts = [payload]
    if dist.is_available() and dist.is_initialized():
        parts = [None] * dist.get_world_size()
        dist.all_gather_object(parts, payload)
    errors = [p["error"] for p in parts if p["error"]]
    if errors:
        raise RuntimeError("Distributed validation failed: " + "; ".join(errors))
    records = [r for part in parts for r in part["records"]]
    if len({r["id"] for r in records}) != len(records):
        raise RuntimeError("Validation shards contain duplicate sample IDs")
    return sorted(records, key=lambda r: r["id"])


@contextlib.contextmanager
def rank_zero_section(trainer):
    """Propagate writer/upload errors instead of leaving other ranks at a barrier."""
    error = None
    try:
        yield trainer.is_global_zero
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    error = trainer.strategy.broadcast(error, src=0)
    if error:
        raise RuntimeError("Validation output failed: " + error)


class TransferValidation(pl.Callback):
    """Owns best/last exports when SEN2NEON is enabled; fallback remains native."""

    def __init__(self, prepared: dict, validation: dict, run_dir: str | Path):
        self.prepared, self.options, self.run_dir = prepared, validation, Path(run_dir)
        self.enabled = bool(prepared["sen2neon_root"])
        self.rows = prepared["sen2neon_rows"]
        ids = set(prepared["selection_ids"])
        self.selection_rows = [r for r in self.rows if r["id"] in ids]
        self.raw_rows = prepared["raw_rows"]
        neon = validation.get("sen2neon", {})
        self.tolerance = float(neon.get("spectral_tolerance", 0.10))
        self.preview_count = int(neon.get("preview_count", 8))
        self.save_all_reconstructions = bool(neon.get("save_all_reconstructions", True))
        if self.preview_count < 0:
            raise ValueError("sen2neon.preview_count must be nonnegative")
        self.every_n_steps = int(validation.get("every_n_train_steps", 500))
        if self.every_n_steps <= 0:
            raise ValueError("validation.every_n_train_steps must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError(
                "sen2neon.spectral_tolerance must be finite and nonnegative"
            )
        self.identity = prepared["request_hash"]
        self.baseline: dict | None = None
        self.best_score = math.inf
        self.best_step = 0
        self.last_evaluated_step = -1
        self.best_evaluation = ""
        self.baseline_evaluation = ""
        self.last_evaluation = ""
        self._lpips = None

    @property
    def state_key(self):
        return "OpenSRTransferValidation/v1"

    def state_dict(self):
        return {
            k: getattr(self, k)
            for k in (
                "identity",
                "baseline",
                "best_score",
                "best_step",
                "last_evaluated_step",
                "best_evaluation",
                "baseline_evaluation",
                "last_evaluation",
            )
        }

    def load_state_dict(self, state_dict):
        if state_dict["identity"] != self.identity:
            raise ValueError(
                "Cannot resume with changed validation data or sampling protocol"
            )
        for key, value in state_dict.items():
            setattr(self, key, value)

    def _barrier(self, trainer):
        trainer.strategy.barrier()

    def _export(self, trainer, module, name, score):
        with rank_zero_section(trainer) as writer:
            if writer:
                path = self.run_dir / "checkpoints/native" / f"{name}-inference.ckpt"
                state = materialize_diffusion_ema_weights(module.native_state_dict())
                save_native_checkpoint(
                    path,
                    state,
                    epoch=int(trainer.current_epoch),
                    global_step=int(trainer.global_step),
                    metadata={
                        "stage": "diffusion",
                        "inference_unet_weight_source": "ema",
                        "monitor": "val/sen2neon/rgb_lpips",
                        "monitor_value": score,
                        "opensr_inference_config": module.opensr_inference_config(),
                        "validation_identity": self.identity,
                        "spectral_tolerance": self.tolerance,
                        "baseline_metrics": self.baseline,
                    },
                )

    def _resume_checkpoint(self, trainer, name):
        path = self.run_dir / "checkpoints/resume" / f"{name}.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.ckpt")
        trainer.save_checkpoint(temporary)
        if trainer.is_global_zero:
            temporary.replace(path)
        self._barrier(trainer)

    def on_fit_start(self, trainer, pl_module):
        with rank_zero_section(trainer) as writer:
            run = wandb_run(trainer)
            if run is not None:
                import wandb
                from omegaconf import OmegaConf

                run.config.update(
                    OmegaConf.to_container(
                        OmegaConf.load(self.run_dir / "resolved_config.yaml"),
                        resolve=True,
                    ),
                    allow_val_change=True,
                )
                artifact = wandb.Artifact(
                    safe_id(f"{run.id}-provenance"), type="training-provenance"
                )
                for name in (
                    "resolved_config.yaml",
                    "data_manifest.json",
                    "provenance.json",
                ):
                    artifact.add_file(str(self.run_dir / name))
                source = self.run_dir / "source"
                if source.exists():
                    artifact.add_dir(str(source))
                run.log_artifact(artifact)
        if self.enabled and self.baseline is None:
            self._check(trainer, pl_module, initial=True)
        elif self.enabled:
            for name in ("baseline", "best"):
                if not (
                    self.run_dir / "checkpoints/native" / f"{name}-inference.ckpt"
                ).is_file():
                    raise FileNotFoundError(
                        f"Resume requires original {name} native checkpoint in run_dir"
                    )
        elif self.raw_rows:
            self._raw_previews(trainer, pl_module, "raw_s2_step00000000")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if trainer.state.fn != TrainerFn.FITTING:
            (
                self._evaluate(trainer, pl_module, self.selection_rows, "validate_only")
                if self.enabled
                else None
            )
            if self.raw_rows:
                self._raw_previews(trainer, pl_module, "raw_s2_validate_only")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # Count optimizer updates directly: accumulation at an odd epoch boundary
        # must not shift the external-validation schedule.
        step = int(trainer.global_step)
        if step == 0 or step == self.last_evaluated_step or step % self.every_n_steps:
            return
        if self.enabled:
            self._check(trainer, pl_module)
        elif self.raw_rows:
            self._raw_previews(trainer, pl_module, f"raw_s2_step{step:08d}")
            self.last_evaluated_step = step

    def _check(self, trainer, module, *, initial=False):
        step = int(trainer.global_step)
        label = f"sen2neon_step{step:08d}"
        records = self._evaluate(trainer, module, self.selection_rows, label)
        scores = aggregate(records)
        # Missing selection metrics on even one tile must not disappear in averaging.
        if any(
            any(r["metrics"].get(k) is None for k in ("rgb_lpips", "nir_mae", "sam"))
            for r in records
        ):
            raise ValueError(
                "Nonfinite SEN2NEON selection metric; no checkpoint selected"
            )
        if initial:
            self.baseline = scores
            self.baseline_evaluation = label
        eligible, reasons = guardrail_decision(scores, self.baseline, self.tolerance)
        improved = eligible and scores["rgb_lpips"] < self.best_score
        self.last_evaluated_step = step
        self.last_evaluation = label
        if improved:
            self.best_score, self.best_step, self.best_evaluation = (
                scores["rgb_lpips"],
                step,
                label,
            )
        metrics = {f"val/sen2neon/{k}": v for k, v in scores.items()}
        metrics.update(
            {
                "val/sen2neon/eligible": float(eligible),
                "val/sen2neon/best_lpips": self.best_score,
            }
        )
        # Every rank receives identical globally aggregated values before save hooks.
        trainer.callback_metrics.update(
            {k: torch.tensor(v, device=module.device) for k, v in metrics.items()}
        )
        with rank_zero_section(trainer) as writer:
            if trainer.is_global_zero and trainer.logger:
                trainer.logger.log_metrics(metrics, step=step)
            if trainer.is_global_zero:
                write_json(
                    self.run_dir / "validation" / label / "selection.json",
                    {
                        "optimizer_step": step,
                        "eligible": eligible,
                        "improved": improved,
                        "rejection_reasons": reasons,
                        "scores": scores,
                        "baseline": self.baseline,
                        "tolerance": self.tolerance,
                        "best_step": self.best_step,
                    },
                )
                print(
                    f"SEN2NEON step={step} LPIPS={scores['rgb_lpips']:.5f} NIR={scores['nir_mae']:.5f} "
                    f"SAM={scores['sam']:.5f} eligible={eligible} best={self.best_step} reasons={reasons}",
                    flush=True,
                )
                run = wandb_run(trainer)
                if run is not None:
                    import wandb

                    artifact = wandb.Artifact(
                        safe_id(f"{run.id}-{label}-selection"),
                        type="checkpoint-selection",
                    )
                    artifact.add_file(
                        str(self.run_dir / "validation" / label / "selection.json")
                    )
                    run.log_artifact(artifact)
        if self.raw_rows:
            self._raw_previews(trainer, module, f"raw_s2_step{step:08d}")
        if initial:
            self._export(trainer, module, "baseline", scores["rgb_lpips"])
        self._export(trainer, module, "last", scores["rgb_lpips"])
        if improved:
            self._export(trainer, module, "best", self.best_score)
            self._resume_checkpoint(trainer, "best")
        self._resume_checkpoint(trainer, "last")

    def _sample(self, trainer, module, lr, sample_id):
        seed = (
            int(hashlib.sha256(sample_id.encode()).hexdigest()[:8], 16)
            + int(self.options.get("seed", 42))
        ) % (2**31)
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if module.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with context:
            sr = module.sample_super_resolution(
                lr,
                steps=int(self.options.get("sampling_steps", 100)),
                eta=float(self.options.get("sampling_eta", 0.95)),
                temperature=float(self.options.get("temperature", 1)),
                seed=seed,
                use_ema=True,
            )
        if not torch.isfinite(sr).all():
            raise ValueError(f"Nonfinite reconstruction for {sample_id}")
        return sr.float()

    def _evaluate(self, trainer, module, rows, label, reuse=None, *, save_all=None):
        output = self.run_dir / "validation" / label
        output.mkdir(parents=True, exist_ok=True)
        previews = (
            select_diverse(rows, min(self.preview_count, len(rows)))
            if self.preview_count and rows
            else []
        )
        preview_ids = {row["id"] for row in previews}
        retain_all = self.save_all_reconstructions if save_all is None else save_all
        reused = {}
        if reuse:
            old = self.run_dir / "validation" / reuse
            source = old / "metrics.json"
            if source.exists():
                reused = {r["id"]: r for r in json.loads(source.read_text())}
        if self._lpips is None:
            import lpips

            # Isolate pretrained-network construction from training's RNG stream.
            with torch.random.fork_rng(devices=[]):
                self._lpips = (
                    lpips.LPIPS(net="vgg", spatial=True).eval().requires_grad_(False)
                )
        self._lpips.to(module.device)
        local, error = [], None
        dataset = SEN2NEONDataset(self.prepared["sen2neon_root"], rows)
        try:
            with validation_inference(module):
                for index in range(trainer.global_rank, len(rows), trainer.world_size):
                    row = rows[index]
                    file = output / f"{safe_id(row['id'])}.npz"
                    retain = retain_all or row["id"] in preview_ids
                    if row["id"] in reused:
                        record = dict(reused[row["id"]])
                        src = (
                            (self.run_dir / "validation" / reuse / record["file"])
                            if record.get("file")
                            else None
                        )
                        if not retain or (src is not None and src.is_file()):
                            if retain and not file.exists():
                                os.link(src, file)
                            record["file"] = file.name if retain else None
                            local.append(record)
                            continue
                    sample = dataset[index]
                    lr, hr, mask = [
                        sample[k].unsqueeze(0).to(module.device)
                        for k in ("LR_image", "image", "valid_mask")
                    ]
                    sr = self._sample(trainer, module, lr, row["id"])
                    metrics = finite_json(
                        reconstruction_scores(sr, hr, lr, mask, self._lpips)
                    )
                    if retain:
                        np.savez_compressed(
                            file, sr=sr[0].cpu().numpy().astype(np.float16)
                        )
                    local.append(
                        {
                            "id": row["id"],
                            "site": row["site"],
                            "landcover": row["landcover"],
                            "metrics": metrics,
                            "file": file.name if retain else None,
                            "clipped_fraction": sample["clipped_fraction"],
                        }
                    )
        except Exception as exc:
            error = f"rank {trainer.global_rank}: {type(exc).__name__}: {exc}"
        finally:
            self._lpips.cpu()
        records = gather_records(local, error)
        if {r["id"] for r in records} != {r["id"] for r in rows}:
            raise RuntimeError("Incomplete validation cohort")
        with rank_zero_section(trainer) as writer:
            if writer:
                write_json(output / "metrics.json", records)
                breakdown = {}
                for kind in ("site", "landcover"):
                    groups = defaultdict(list)
                    for row in records:
                        groups[row[kind]].append(row)
                    breakdown[kind] = {
                        key: aggregate(group) for key, group in groups.items()
                    }
                write_json(
                    output / "summary.json",
                    {
                        "mean": aggregate(records),
                        **breakdown,
                        "identity": self.identity,
                        "step": int(trainer.global_step),
                        "retained_reconstructions": sum(
                            r["file"] is not None for r in records
                        ),
                    },
                )
                table = []
                for row in previews:
                    sample = dataset[rows.index(row)]
                    sr = np.load(output / f"{safe_id(row['id'])}.npz")["sr"].astype(
                        np.float32
                    )
                    table.append(
                        preview_row(
                            output / "previews",
                            row,
                            sample["LR_image"].numpy(),
                            sr,
                            sample["image"].numpy(),
                        )
                    )
                log_outputs(
                    trainer, output, table, label=label, step=int(trainer.global_step)
                )
        self._barrier(trainer)
        return records

    def _raw_previews(self, trainer, module, label):
        output = self.run_dir / "validation" / label
        output.mkdir(parents=True, exist_ok=True)
        local, error = [], None
        try:
            with validation_inference(module):
                for row in self.raw_rows[trainer.global_rank :: trainer.world_size]:
                    lr = read_raw(row)
                    padded, (h, w) = pad_lr(lr.unsqueeze(0).to(module.device))
                    sr = (
                        self._sample(trainer, module, padded, row["id"])[0, :, :h, :w]
                        .cpu()
                        .numpy()
                    )
                    file = output / f"{safe_id(row['id'])}.npz"
                    np.savez_compressed(file, sr=sr.astype(np.float16))
                    local.append({"id": row["id"], "file": file.name})
        except Exception as exc:
            error = f"rank {trainer.global_rank}: {type(exc).__name__}: {exc}"
        records = gather_records(local, error)
        with rank_zero_section(trainer) as writer:
            if writer:
                write_json(
                    output / "manifest.json",
                    {
                        "rows": self.raw_rows,
                        "outputs": records,
                        "identity": self.identity,
                    },
                )
                table = []
                for row in self.raw_rows:
                    sr = np.load(output / f"{safe_id(row['id'])}.npz")["sr"].astype(
                        np.float32
                    )
                    table.append(
                        preview_row(output / "previews", row, read_raw(row).numpy(), sr)
                    )
                log_outputs(
                    trainer, output, table, label=label, step=int(trainer.global_step)
                )
        self._barrier(trainer)

    def on_train_end(self, trainer, pl_module):
        if not self.enabled:
            return
        if self.last_evaluated_step != int(trainer.global_step):
            self._check(trainer, pl_module)
        # Save current loop state even when the last check occurred at the same step.
        self._resume_checkpoint(trainer, "last")
        if not self.options.get("sen2neon", {}).get("final_assessment", True):
            return
        results = {}
        try:
            for name, reuse in (
                ("baseline", self.baseline_evaluation),
                ("best", self.best_evaluation),
                ("last", self.last_evaluation),
            ):
                checkpoint = (
                    self.run_dir / "checkpoints/native" / f"{name}-inference.ckpt"
                )
                load_diffusion_weights(pl_module.diffusion, checkpoint)
                # A resumed run can finish again at a later step. Keep prior
                # artifact paths immutable and never pair new metrics with old files.
                label = f"final_step{int(trainer.global_step):08d}_{name}"
                results[name] = self._evaluate(
                    trainer, pl_module, self.rows, label, reuse=reuse, save_all=True
                )
            with rank_zero_section(trainer) as writer:
                if writer:
                    baseline = aggregate(results["baseline"])
                    report = {
                        "selection_tiles": len(self.selection_rows),
                        "assessment_tiles": len(self.rows),
                        "best_selection_step": self.best_step,
                        "baseline": baseline,
                        "candidates": {},
                    }
                    for name in ("best", "last"):
                        scores = aggregate(results[name])
                        eligible, reasons = guardrail_decision(
                            scores, baseline, self.tolerance
                        )
                        report["candidates"][name] = {
                            "scores": scores,
                            "spectral_limits_pass": eligible,
                            "reasons": reasons,
                            "paired_changes": paired_report(
                                results["baseline"], results[name]
                            ),
                        }
                    write_json(self.run_dir / "validation/final_report.json", report)
                    run = wandb_run(trainer)
                    if run is not None:
                        import wandb

                        artifact = wandb.Artifact(
                            safe_id(f"{run.id}-final-assessment"), type="evaluation"
                        )
                        artifact.add_file(
                            str(self.run_dir / "validation/final_report.json")
                        )
                        run.log_artifact(artifact)
        finally:
            load_diffusion_weights(
                pl_module.diffusion, self.run_dir / "checkpoints/resume/last.ckpt"
            )
            self._barrier(trainer)
