"""Record executable sources and dependency/checkpoint identities with each run."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

from ..validation.data import digest_file, digest_json, preparation_lock, write_json


def record_provenance(config, run_dir: Path):
    package = Path(__file__).resolve().parents[2]
    sources = sorted(p for p in package.rglob("*") if p.is_file() and p.suffix in {".py", ".yaml", ".json"})
    hashes = {str(p.relative_to(package)): digest_file(p) for p in sources}
    with preparation_lock(run_dir / ".provenance.lock"):
        path = run_dir / "provenance.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["source_hash"] != digest_json(hashes):
                raise ValueError("Executable sources changed since run creation; use a new run_dir")
            return
        versions = {}
        for name in ("torch", "pytorch-lightning", "tacoreader", "rasterio", "lpips", "torchmetrics", "torchvision", "wandb", "numpy"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        try:
            revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package, text=True, stderr=subprocess.DEVNULL).strip()
            dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=package, text=True))
        except (OSError, subprocess.CalledProcessError):
            revision, dirty = None, None
        checkpoint = config.model.get("pretrained_checkpoint")
        for source in sources:
            target = run_dir / "source" / source.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        write_json(path, {
            "git_revision": revision, "git_dirty": dirty, "source_hash": digest_json(hashes),
            "sources": hashes, "python": sys.version, "dependencies": versions,
            "initial_checkpoint": str(checkpoint) if checkpoint else None,
            "initial_checkpoint_sha256": digest_file(checkpoint) if checkpoint else None,
            "seed": int(config.seed),
        })
