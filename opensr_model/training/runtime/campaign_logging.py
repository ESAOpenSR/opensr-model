"""Durable local event log consumed by the campaign's sole W&B publisher."""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import torch
from pytorch_lightning import Callback
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.utilities import rank_zero_only


class CampaignLogger(Logger):
    def __init__(self, run_dir):
        super().__init__()
        self.directory = Path(run_dir)
        self.directory.mkdir(parents=True, exist_ok=True)

    @property
    def name(self):
        return "campaign"

    @property
    def version(self):
        return "0"

    @property
    def log_dir(self):
        return str(self.directory)

    @rank_zero_only
    def log_hyperparams(self, params):
        pass  # The immutable resolved configuration is uploaded as an artifact.

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        values = {}
        for key, value in metrics.items():
            value = float(value)
            values[key] = value if math.isfinite(value) else None
        event = {"step": step, "time": time.time(), "metrics": values}
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @rank_zero_only
    def finalize(self, status):
        self.save()


class SourceExposure(Callback):
    """Checkpoint actual exposures, and reject nonfinite training losses."""

    def __init__(self):
        self.counts = {}
        self.last_logged = -1

    def state_dict(self):
        return {"counts": self.counts, "last_logged": self.last_logged}

    def load_state_dict(self, state_dict):
        self.counts = dict(state_dict["counts"])
        self.last_logged = state_dict["last_logged"]

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if loss is not None and not torch.isfinite(loss).all():
            raise ValueError("Nonfinite training loss; campaign branch stopped")
        names = batch.get("source", ["observed"] * len(batch["image"]))
        for name in names:
            self.counts[name] = self.counts.get(name, 0) + 1
        step = int(trainer.global_step)
        if step != self.last_logged and step % 20 == 0:
            # Exact fixed per-rank source quotas make local counts equal on all ranks.
            if trainer.is_global_zero and trainer.logger:
                trainer.logger.log_metrics(
                    {
                        f"exposure/{k}": float(v * trainer.world_size)
                        for k, v in self.counts.items()
                    },
                    step=step,
                )
            self.last_logged = step

    def on_train_end(self, trainer, module):
        if trainer.is_global_zero and trainer.logger:
            trainer.logger.log_metrics(
                {
                    f"exposure/{k}": float(v * trainer.world_size)
                    for k, v in self.counts.items()
                },
                step=int(trainer.global_step),
            )
