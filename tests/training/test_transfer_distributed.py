"""Small real process-group test: uneven/empty shards and propagated errors."""
import json
from pathlib import Path

import torch.distributed as dist
import torch.multiprocessing as mp

from opensr_model.training.validation.metrics import aggregate
from opensr_model.training.validation.callback import gather_records


def _worker(rank, directory):
    dist.init_process_group("gloo", init_method=f"file://{directory}/init", rank=rank, world_size=2)
    try:
        results = []
        for size in (5, 1):
            local = [{"id": str(i), "metrics": {"mae": float(i)}} for i in range(rank, size, 2)]
            records = gather_records(local)
            results.append({"count": len(records), "mean": aggregate(records)["mae"]})
        try:
            gather_records([], "simulated failure" if rank == 0 else None)
        except RuntimeError as exc:
            assert "simulated failure" in str(exc)
        else:
            raise AssertionError("Failure was not propagated")
        Path(directory, f"{rank}.json").write_text(json.dumps(results))
    finally:
        dist.destroy_process_group()


def test_uneven_distributed_validation_matches_single_process(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path),), nprocs=2, join=True)
    expected = [{"count": 5, "mean": 2.0}, {"count": 1, "mean": 0.0}]
    for rank in range(2):
        assert json.loads((tmp_path / f"{rank}.json").read_text()) == expected
