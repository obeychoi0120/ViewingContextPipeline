"""Synthetic mean/GCN training cost; no BGE, Run, profiler or shared-cache access."""

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "validation"))

from graph_fixture import synthetic_store  # noqa: E402
from validation.graph_model import GraphSASRec  # noqa: E402
from validation.model import torch, seed_everything  # noqa: E402
from validation.leave_one_out_execution import negative_mask  # noqa: E402


def model(store, aggregation):
    return GraphSASRec(store=store, aggregation=aggregation, item_count=len(store),
        max_length=10, embedding_dim=512, num_blocks=2, num_heads=2, dropout=0,
        arm="graph")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(store, aggregation, common, sequences, targets, args, device):
    seed_everything(42)
    net = model(store, aggregation)
    missing, extra = net.load_state_dict(common, strict=False)
    assert not extra and all(name.startswith("graph_encoder.gcn.") for name in missing)
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.1)
    ids = torch.cat([sequences.flatten(), targets])
    timings = []
    for step in range(args.warmup + args.steps):
        if step == args.warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        started = time.perf_counter()
        net.train()
        optimizer.zero_grad(set_to_none=True)
        net.prepare_items(ids)
        logits = net.user_vectors(sequences) @ net.item_vectors(targets).T
        logits = logits.masked_fill(negative_mask(sequences, targets), -1e4)
        loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(targets), device=device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        net.clear_item_cache()
        sync(device)
        if step >= args.warmup:
            timings.append(time.perf_counter() - started)
    result = {
        "median_batch_seconds": statistics.median(timings),
        "batch_seconds": timings,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 1024 ** 2
            if device.type == "cuda" else None,
    }
    del loss, logits, optimizer, net
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--videos", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.videos, args.batch_size, args.steps, args.repeats) < 1 or args.warmup < 0:
        parser.error("counts must be positive; warmup must be nonnegative")
    torch.set_num_threads(1)
    seed_everything(42)
    device = torch.device(args.device)
    store = synthetic_store(args.videos)
    common = model(store, "mean").state_dict()
    rng = np.random.default_rng(42)
    sequences = torch.tensor(rng.integers(1, args.videos + 1, size=(args.batch_size, 10)), device=device)
    targets = torch.tensor(rng.integers(1, args.videos + 1, size=args.batch_size), device=device)
    measurements = {"mean": [], "gnn": []}
    for repeat in range(args.repeats):
        for aggregation in (("mean", "gnn") if repeat % 2 == 0 else ("gnn", "mean")):
            measurements[aggregation].append(measure(
                store, aggregation, common, sequences, targets, args, device))
    result = {
        "synthetic_only": True, "profiler": False, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "seed": 42, "videos": args.videos,
        "batch_size": args.batch_size, "feature_dim": 1024, "item_dim": 512,
        "scenes": len(store["contexts"]),
        "scene_pairs": int(np.square(np.diff(store["video_offsets"])).sum()),
        "warmup": args.warmup, "steps": args.steps, "repeats": args.repeats,
        "measurements": measurements,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
