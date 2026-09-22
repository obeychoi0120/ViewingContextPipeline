"""Independent day/seed/arm selection, refit and atomic combination resume."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from pipeline_runtime import read_json, write_json
from validation.metrics import metrics_from_rank
from validation.model import require_torch, save_checkpoint, seed_everything, torch
from validation.representation_checks import verify_representations
from validation.recommendation import _new_model, _optimizer
from validation.recommendation_contracts import ARCHITECTURE_VERSION, resolve_target_arms
from validation.rolling_data import EventTable, iter_jsonl
from validation.selection import load_validation_cohort, training_signature
from validation.rolling_execution import (
    EXECUTION_VERSION, execution_for, masked_ranks, negative_mask,
)

SCHEMA = "sasrec-rolling-combination/v3"
LEGACY_SCHEMA = "sasrec-rolling-combination/v1"


def phase_ids(table, split, phase):
    bounds = split["phases"][phase]
    return table.select(bounds["start_ms"], bounds["end_ms"])


def transition_loss(model, table, ids, probabilities, device):
    execution = execution_for(table)
    target_ids = table.targets[ids]
    targets = torch.as_tensor(target_ids, dtype=torch.long, device=device)
    inputs = execution.inputs(ids, model.max_length, device)
    users = model.user_vectors(inputs)
    logits = users @ model.item_vectors(targets).T
    logs = execution.log_probabilities(probabilities, target_ids, device, logits.dtype)
    logits = logits - logs[targets][None, :]
    masks = negative_mask(inputs, targets)
    logits = logits.masked_fill(masks, -1e4)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ids), device=device))
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite transition loss")
    return loss


def train_epoch(model, optimizer, table, ids, probabilities, rng, config, device):
    model.train()
    total = torch.zeros((), dtype=torch.float64, device=device)
    updates = 0
    order = rng.permutation(ids)
    for start in range(0, len(order), config.model.batch_size):
        batch = order[start : start + config.model.batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = transition_loss(model, table, batch, probabilities, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.detach().to(torch.float64) * len(batch)
        updates += 1
    if not len(ids):
        raise RuntimeError("empty training partition")
    return {"loss": float(total.cpu()) / len(ids), "positive_count": len(ids), "optimizer_updates": updates}


def rank_batches(model, table, ids, config, device):
    """Share ranking between validation-only and full per-event evaluation."""
    model.eval()
    execution = execution_for(table)
    with torch.no_grad():
        catalog = model.catalog_vectors()
        for start in range(0, len(ids), config.model.batch_size):
            batch = ids[start : start + config.model.batch_size]
            inputs = execution.inputs(batch, model.max_length, device)
            scores = model.user_vectors(inputs) @ catalog.T
            ranks = masked_ranks(scores, table, batch).cpu().tolist()
            yield batch, ranks


def validation_ndcg10(model, table, ids, config, device):
    # Python float/math.log2 and source order match the previous metric reduction.
    return sum(
        1.0 / math.log2(rank + 1) if rank <= 10 else 0.0
        for _, ranks in rank_batches(model, table, ids, config, device)
        for rank in ranks
    ) / len(ids)


def evaluate(model, table, ids, config, device):
    """Frozen parameters, cached catalog vectors, strictly earlier full seen mask."""
    for batch, ranks in rank_batches(model, table, ids, config, device):
        for event, rank in zip(batch, ranks, strict=True):
            yield {
                **table.rows[int(event)],
                "rank": rank,
                "candidate_count": len(table.items),
                **metrics_from_rank(rank, config.evaluation.cutoffs),
            }


def combination_dir(context, date, seed, arm):
    return context.recommendations_dir / date / f"seed_{seed}" / arm.lower()


def combination_complete(directory, identity, expected_count, *, architecture_version=None):
    # Resume always requires the current architecture unless a read-only caller
    # explicitly selects a supported historical version.
    read_only = architecture_version is not None
    if architecture_version is None:
        architecture_version = ARCHITECTURE_VERSION
    try:
        complete = read_json(directory / "complete.json")
        schema = complete.get("schema_version")
        if schema not in ({SCHEMA, LEGACY_SCHEMA, "sasrec-rolling-combination/v2"} if read_only else {SCHEMA}) or any(
            complete.get("identity", {}).get(key) != value for key, value in identity.items()
        ):
            return False
        if complete.get("event_count") != expected_count:
            return False
        names = {"sasrec.pt", "training.json", "per_event_metrics.jsonl"}
        if not all((directory / name).is_file() and (directory / name).stat().st_size for name in names):
            return False
        if schema in {SCHEMA, "sasrec-rolling-combination/v2"}:
            from validation.shared_cache import checksum
            if set(complete.get("checksums", {})) != names or any(
                checksum(directory / name) != complete["checksums"][name] for name in names
            ):
                return False
        training = read_json(directory / "training.json")
        if training.get("architecture_version") != architecture_version:
            return False
        if training.get("schema_version") != schema or any(
            training.get(key) != value for key, value in identity.items()
        ):
            return False
        if not training.get("selection") or len(training.get("refit", [])) != training.get("best_epoch"):
            return False
        seen = set()
        for row in iter_jsonl(directory / "per_event_metrics.jsonl"):
            event = row["event_id"]
            if (
                row.get("schema_version") != "sasrec-per-event-metrics/v1"
                or type(event) is not int
                or event < 0
                or event in seen
                or any(row.get(key) != value for key, value in identity.items())
                or type(row["rank"]) is not int
                or not 1 <= row["rank"] <= training["catalog_size"]
                or row["candidate_count"] != training["catalog_size"]
            ):
                return False
            seen.add(event)
        return len(seen) == expected_count
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def run_combination(context, config, table, split, identity, branch, prepared, device):
    ids, probabilities, frequencies = prepared
    date, seed, arm = (identity[key] for key in ("evaluation_date", "seed", "arm"))
    directory = combination_dir(context, date, seed, arm)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "complete.json").unlink(missing_ok=True)
    print(f"[Rolling] {date} seed={seed} {arm}: selection device={device}", flush=True)
    started = time.monotonic()
    timings = dict.fromkeys(("preparation", "selection_training", "validation", "refit", "test"), 0.0)
    execution = execution_for(table)
    execution.padded(config.model.max_sequence_length)
    for phase in ("selection", "refit"):
        execution.log_probabilities(probabilities[phase], table.targets[ids[phase]], device, torch.float32)
    timings["preparation"] = time.monotonic() - started
    with np.load(context.representations_dir / f"{branch}_embeddings.npz") as data:
        features = data["values"]
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = _new_model(
        config,
        item_count=len(table.items),
        branch=branch,
        features=features,
        device=device,
    )
    optimizer = _optimizer(model, config)
    selection = []
    best_epoch, best_score = 0, -math.inf
    for epoch in range(1, config.model.max_epochs + 1):
        phase_started = time.monotonic()
        record = train_epoch(
            model,
            optimizer,
            table,
            ids["selection"],
            probabilities["selection"],
            rng,
            config,
            device,
        )
        timings["selection_training"] += time.monotonic() - phase_started
        phase_started = time.monotonic()
        score = validation_ndcg10(model, table, ids["validation"], config, device)
        timings["validation"] += time.monotonic() - phase_started
        selection.append({"epoch": epoch, **record, "validation_ndcg10": score})
        print(
            f"[Rolling] {date} {arm} seed={seed} epoch={epoch} valid={score:.6f}",
            flush=True,
        )
        if score > best_score:
            best_epoch, best_score = epoch, score
        if epoch - best_epoch >= config.model.patience:
            break
    del model, optimizer
    seed_everything(seed)
    rng = np.random.default_rng(seed)
    model = _new_model(
        config,
        item_count=len(table.items),
        branch=branch,
        features=features,
        device=device,
    )
    optimizer = _optimizer(model, config)
    print(f"[Rolling] {date} {arm} seed={seed} refit epochs={best_epoch} device={device}", flush=True)
    refit = []
    phase_started = time.monotonic()
    for epoch in range(1, best_epoch + 1):
        refit.append(
            {
                "epoch": epoch,
                **train_epoch(
                    model,
                    optimizer,
                    table,
                    ids["refit"],
                    probabilities["refit"],
                    rng,
                    config,
                    device,
                ),
            }
        )
    timings["refit"] = time.monotonic() - phase_started
    phase_started = time.monotonic()
    print(f"[Rolling] {date} {arm} seed={seed} test device={device}", flush=True)
    temporary = directory / "per_event_metrics.jsonl.tmp"
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in evaluate(model, table, ids["test"], config, device):
            row.update(identity)
            row["schema_version"] = "sasrec-per-event-metrics/v1"
            row["refit_item_frequency"] = int(
                frequencies[table.targets[row["event_id"]]]
            )
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if count != len(ids["test"]):
        raise RuntimeError("incomplete evaluation")
    temporary.replace(directory / "per_event_metrics.jsonl")
    timings["test"] = time.monotonic() - phase_started
    metadata = {
        **identity,
        "architecture_version": ARCHITECTURE_VERSION,
        "best_epoch": best_epoch,
        "catalog_size": len(table.items),
    }
    save_checkpoint(directory / "sasrec.pt", model, metadata)
    write_json(
        directory / "training.json",
        {
            "schema_version": SCHEMA,
            **metadata,
            "split": split,
            "selection": selection,
            "refit": refit,
            "untrained_test_target_fraction": float(
                np.mean(frequencies[table.targets[ids["test"]]] == 0)
            ),
            "elapsed_seconds": time.monotonic() - started,
            "execution": {"version": EXECUTION_VERSION, "seconds": timings},
            "device": str(device),
            "environment": {
                "python": sys.version,
                "torch": str(torch.__version__),
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
            },
        },
    )
    from validation.shared_cache import checksum
    write_json(
        directory / "complete.json",
        {
            "schema_version": SCHEMA,
            "identity": identity,
            "event_count": count,
            "checksums": {name: checksum(directory / name) for name in (
                "sasrec.pt", "training.json", "per_event_metrics.jsonl")},
        },
    )
    # Publish each completed unit, including in spawned workers, so interruption
    # of a later combination does not withhold already finished shared results.
    from validation import recommendation_cache
    if recommendation_cache.eligible(context, branch):
        recommendation_cache.publish(context, directory, identity, table, split, force=True)


def prepare_split(table, split):
    ids = {phase: phase_ids(table, split, phase) for phase in split["phases"]}
    probabilities = {}
    for phase in ("selection", "refit"):
        counts = np.bincount(table.targets[ids[phase]], minlength=len(table.items) + 1)
        probabilities[phase] = counts / counts.sum()
    raw_refit = table.select(end=split["phases"]["refit"]["end_ms"], eligible=False)
    frequencies = np.bincount(table.targets[raw_refit], minlength=len(table.items) + 1)
    return ids, probabilities, frequencies


def worker_devices(workers_per_gpu):
    if type(workers_per_gpu) is not int or workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be a positive integer")
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not available:
        if workers_per_gpu == 1:
            return ["cpu"]
        raise ValueError("workers_per_gpu requires visible CUDA devices")
    # Logical indices respect CUDA_VISIBLE_DEVICES, including UUID masks.
    # Round-robin order spreads a small pending set across visible GPUs first.
    return [f"cuda:{i}" for _ in range(workers_per_gpu) for i in range(available)]


def run_rolling(context, *, force=False, workers_per_gpu=1, target=None):
    from validation.steps import validation_config
    from validation.representation_provenance import recommendation_identity

    arms = resolve_target_arms(target, config=context.config)
    from validation import recommendation_cache
    from validation.selection import LEGACY_POLICY

    config = validation_config(context)
    cohort = load_validation_cohort(context)
    if cohort["manifest"]["policy"] == LEGACY_POLICY:
        raise ValueError("run embed-representations to create the full-catalog cohort first")
    table = EventTable(cohort["events"])
    verify_representations(context, cohort, arms=arms)
    training_input_hash = training_signature(context, cohort, config)
    completed = skipped = shared = 0
    jobs = []
    combinations = []
    for split in cohort["plan"]["splits"]:
        expected_count = len(phase_ids(table, split, "test"))
        for seed in config.model.seeds:
            for arm, branch in arms.items():
                identity = {
                    "run_id": context.run_id,
                    "evaluation_date": split["evaluation_date"],
                    "seed": seed,
                    "arm": arm,
                    "training_input_hash": training_input_hash,
                    **recommendation_identity(context, branch),
                }
                directory = combination_dir(context, split["evaluation_date"], seed, arm)
                combinations.append((directory, identity, branch, split))
                if not force and combination_complete(
                    directory, identity, expected_count
                ):
                    skipped += 1
                elif (not force and recommendation_cache.eligible(context, branch)
                      and recommendation_cache.restore(context, directory, identity, table, split)):
                    shared += 1
                    skipped += 1
                else:
                    jobs.append((split, identity, branch))
    devices = []
    if jobs:
        require_torch()
        devices = worker_devices(workers_per_gpu)
    total = skipped + len(jobs)
    print(
        f"[Rolling] pending={len(jobs)} reused={skipped} "
        f"workers={min(len(devices), len(jobs))} devices={','.join(dict.fromkeys(devices))}",
        flush=True,
    )
    with tqdm(
        total=total, initial=skipped, desc="Rolling recommendation", unit="run", file=sys.stdout
    ) as progress:
        progress.set_postfix(reused=skipped)
        if jobs and len(devices) > 1:
            from validation.rolling_workers import run_parallel
            completed = run_parallel(context, jobs, devices, progress)
        elif jobs:
            previous_date, prepared = None, None
            device = torch.device(devices[0])
            for split, identity, branch in jobs:
                if split["evaluation_date"] != previous_date:
                    prepared = prepare_split(table, split)
                    previous_date = split["evaluation_date"]
                progress.set_postfix(
                    date=previous_date, seed=identity["seed"], arm=identity["arm"], reused=skipped
                )
                run_combination(context, config, table, split, identity, branch, prepared, device)
                completed += 1
                progress.update(1)
    for directory, identity, branch, split in combinations:
        if recommendation_cache.eligible(context, branch):
            recommendation_cache.publish(context, directory, identity, table, split, force=force)
    write_json(context.recommendations_dir / "reuse.json", {
        "run_id": context.run_id, "local": skipped - shared, "shared": shared, "generated": completed,
    })
    print(f"[Rolling] completed={completed} skipped={skipped}", flush=True)
    return {"stage": "run-recommendation", "completed": completed, "skipped": skipped}
