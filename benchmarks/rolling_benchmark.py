"""Isolated rolling kernels benchmark; never opens a Run or shared cache."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "validation"))

import rolling_reference as reference  # noqa: E402
from validation import rolling_recommendation as optimized  # noqa: E402
from validation.model import SASRec, seed_everything, torch  # noqa: E402
from validation.rolling_data import EventTable  # noqa: E402
from validation.rolling_execution import execution_for  # noqa: E402


def records(items, users, history):
    # Single-event users establish the catalog without quadratic filler histories.
    rows = [dict(user_id=f"catalog-{i}", item_id=str(i + 1), timestamp=0)
            for i in range(items)]
    rng = np.random.default_rng(31)
    for user in range(users):
        for event in range(history):
            rows.append(dict(user_id=f"user-{user}", item_id=str(rng.integers(1, items + 1)),
                             timestamp=event + 1))
    return [dict(event_id=i, **row) for i, row in enumerate(rows)]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(implementation, rows, features, device):
    seed_everything(42)
    config = SimpleNamespace(model=SimpleNamespace(batch_size=256),
                             evaluation=SimpleNamespace(cutoffs=[4, 8, 10, 20, 30]))
    model = SASRec(len(features), 10, 512, 2, 2, 0.1,
                   arm="graph", item_features=features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    table = EventTable(rows)
    ids = table.select()
    counts = np.bincount(table.targets[ids], minlength=len(table.items) + 1)
    probabilities = counts / counts.sum()
    if implementation is optimized:
        runtime = execution_for(table)
        runtime.padded(10)
        runtime.log_probabilities(probabilities, table.targets[ids], device, torch.float32)
    synchronize(device)
    prepared = time.perf_counter()
    trained = implementation.train_epoch(model, optimizer, table, ids, probabilities,
                                          np.random.default_rng(42), config, device)
    synchronize(device)
    training_end = time.perf_counter()
    if implementation is optimized:
        validation = optimized.validation_ndcg10(model, table, ids, config, device)
    else:
        validation = sum(r["NDCG@10"] for r in reference.evaluate(model, table, ids, config, device)) / len(ids)
    synchronize(device)
    validation_end = time.perf_counter()
    result = list(implementation.evaluate(model, table, ids, config, device))
    synchronize(device)
    end = time.perf_counter()
    report = {
        "preparation_seconds": prepared - started,
        "training_seconds": training_end - prepared,
        "validation_seconds": validation_end - training_end,
        "test_seconds": end - validation_end,
        "total_seconds": end - started,
        "training_events_per_second": len(ids) / (training_end - prepared),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "loss": trained["loss"], "validation_ndcg10": validation,
        "event_count": len(ids),
    }
    ranks = [r["rank"] for r in result]
    del model, optimizer, table
    gc.collect()
    return report, ranks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Explicit device, e.g. cpu or cuda:0")
    parser.add_argument("--items", type=int, default=19738)
    parser.add_argument("--users", type=int, default=64)
    parser.add_argument("--history", type=int, default=33)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.items < 2 or args.users < 1 or args.history < 2 or args.repeats < 1:
        parser.error("items/history must be >= 2; users/repeats must be positive")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    rows = records(args.items, args.users, args.history)
    features = np.random.default_rng(17).normal(size=(args.items, 1024)).astype(np.float32)
    features[::7] = 0
    measurements = {"reference": [], "optimized": []}
    rank_reference = None
    # One warmup per implementation, then alternating order to reduce timing bias.
    for repeat in range(args.repeats + 1):
        order = [("reference", reference), ("optimized", optimized)]
        if repeat % 2:
            order.reverse()
        for name, implementation in order:
            report, ranks = measure(implementation, rows, features, device)
            if rank_reference is None:
                rank_reference = ranks
            if ranks != rank_reference:
                raise RuntimeError("benchmark rank parity failed")
            if repeat:
                measurements[name].append(report)
            print(json.dumps({"repeat": repeat, "implementation": name, **report}), flush=True)
    for old, new in zip(measurements["reference"], measurements["optimized"], strict=True):
        if abs(old["loss"] - new["loss"]) > 1e-6 + 1e-5 * abs(old["loss"]):
            raise RuntimeError("benchmark loss tolerance failed")
        if old["validation_ndcg10"] != new["validation_ndcg10"]:
            raise RuntimeError("benchmark validation parity failed")
    medians = {name: {key: statistics.median(r[key] for r in samples) for key in samples[0]}
               for name, samples in measurements.items()}
    document = {
        "environment": {"python": sys.version, "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "device": str(device),
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "measurements": measurements, "median": medians,
        "total_speedup": medians["reference"]["total_seconds"] / medians["optimized"]["total_seconds"],
        "rank_parity": "exact",
        "scope": "Synthetic full-width catalog, one epoch, no Run/cache/checkpoint writes; not full training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"median": medians, "total_speedup": document["total_speedup"]}, indent=2))


if __name__ == "__main__":
    main()
