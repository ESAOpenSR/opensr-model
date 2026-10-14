from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from deployment.opensr_hpc.config import EnvironmentConfig, SlurmConfig
from deployment.opensr_hpc.manifests import write_json


@dataclass(slots=True)
class SlurmJobSpec:
    job_name: str
    script_path: Path
    manifest_path: Path
    output_path: Path
    error_path: Path
    slurm: SlurmConfig
    environment: EnvironmentConfig
    array: str | None = None


def build_sbatch_command(spec: SlurmJobSpec) -> list[str]:
    exports = ["ALL", f"OPENSR_HPC_PYTHON={spec.environment.python_executable}"]
    if spec.environment.modules:
        exports.append(f"OPENSR_HPC_MODULES={','.join(spec.environment.modules)}")
    if spec.environment.conda_env:
        exports.append(f"OPENSR_HPC_CONDA_ENV={spec.environment.conda_env}")

    cmd = [
        "sbatch",
        f"--job-name={spec.job_name}",
        f"--output={spec.output_path}",
        f"--error={spec.error_path}",
        f"--export={','.join(exports)}",
        f"--cpus-per-task={spec.slurm.cpus_per_task}",
        f"--mem={spec.slurm.mem_gb}G",
        f"--time={spec.slurm.time}",
    ]
    if spec.slurm.partition:
        cmd.append(f"--partition={spec.slurm.partition}")
    if spec.slurm.gres:
        cmd.append(f"--gres={spec.slurm.gres}")
    elif spec.slurm.gpus:
        if spec.slurm.gpu_type:
            cmd.append(f"--gpus={spec.slurm.gpu_type}:{spec.slurm.gpus}")
        else:
            cmd.append(f"--gpus={spec.slurm.gpus}")
    if spec.slurm.account:
        cmd.append(f"--account={spec.slurm.account}")
    if spec.slurm.qos:
        cmd.append(f"--qos={spec.slurm.qos}")
    if spec.array:
        cmd.append(f"--array={spec.array}")
    cmd.extend(spec.slurm.extra_args)
    cmd.append(str(spec.script_path))
    cmd.append(str(spec.manifest_path))
    return cmd


def parse_job_id(stdout: str) -> str:
    parts = stdout.strip().split()
    if not parts:
        raise ValueError("Could not parse sbatch output")
    return parts[-1]


def submit_job(
    spec: SlurmJobSpec, submission_dir: Path, dry_run: bool = False
) -> dict[str, str]:
    cmd = build_sbatch_command(spec)
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "sbatch_command.txt").write_text(
        " ".join(cmd) + "\n", encoding="utf-8"
    )

    if dry_run:
        payload = {"mode": "dry-run", "command": " ".join(cmd)}
        write_json(submission_dir / "slurm_job_ids.json", payload)
        return payload

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        payload = {
            "command": " ".join(cmd),
            "returncode": str(exc.returncode),
            "stdout": (exc.stdout or "").strip(),
            "stderr": (exc.stderr or "").strip(),
        }
        write_json(submission_dir / "slurm_job_ids.json", {"mode": "error", **payload})
        stderr = payload["stderr"] or "<empty stderr>"
        stdout = payload["stdout"] or "<empty stdout>"
        raise RuntimeError(
            "sbatch submission failed. "
            f"stderr: {stderr}. stdout: {stdout}. "
            f"Command saved to {submission_dir / 'sbatch_command.txt'}"
        ) from exc

    job_id = parse_job_id(result.stdout)
    payload = {
        "job_id": job_id,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    write_json(submission_dir / "slurm_job_ids.json", payload)
    return payload
