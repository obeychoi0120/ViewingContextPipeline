"""Independent day/seed/arm selection, refit and atomic combination resume."""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

from pipeline_runtime import read_json, write_json
from validation.metrics import metrics_from_rank
from validation.model import pad_sequences, require_torch, save_checkpoint, seed_everything, torch
from validation.representation_checks import verify_representations
from validation.recommendation import _new_model, _optimizer
from validation.recommendation_contracts import ARCHITECTURE_VERSION, RECOMMENDATION_ARMS
from validation.rolling_data import EventTable, iter_jsonl
from validation.scoring import mask_history, rank_of_target

SCHEMA = "sasrec-rolling-combination/v1"


def phase_ids(table, split, phase):
    bounds = split["phases"][phase]
    return table.select(bounds["start_ms"], bounds["end_ms"])


def transition_loss(model, table, ids, probabilities, device):
    histories = [table.history(int(i), model.max_length) for i in ids]
    targets = torch.as_tensor(table.targets[ids], dtype=torch.long, device=device)
    users = model.user_vectors(pad_sequences(histories, model.max_length, device))
    logits = users @ model.item_vectors(targets).T
    probs = torch.as_tensor(probabilities, dtype=logits.dtype, device=device)[targets]
    if not torch.isfinite(probs).all() or probs.le(0).any():
        raise RuntimeError("invalid positive popularity probabilities")
    logits = logits - probs.log()[None, :]
    candidates = targets.tolist()
    masks = []
    for position, event in enumerate(ids):
        observed = set(histories[position]) | {candidates[position]}
        row = [item in observed for item in candidates]
        row[position] = False
        masks.append(row)
    logits = logits.masked_fill(torch.tensor(masks, device=device), -1e4)
    loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ids), device=device))
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite transition loss")
    return loss


def train_epoch(model, optimizer, table, ids, probabilities, rng, config, device):
    model.train()
    total = 0.0
    updates = 0
    order = rng.permutation(ids)
    for start in range(0, len(order), config.model.batch_size):
        batch = order[start : start + config.model.batch_size]
        optimizer.zero_grad(set_to_none=True)
        loss = transition_loss(model, table, batch, probabilities, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * len(batch)
        updates += 1
    if not len(ids):
        raise RuntimeError("empty training partition")
    return {"loss": total / len(ids), "positive_count": len(ids), "optimizer_updates": updates}


def evaluate(model, table, ids, config, device):
    """Frozen parameters, cached catalog vectors, strictly earlier full seen mask."""
    model.eval()
    with torch.no_grad():
        catalog = model.catalog_vectors()
        for start in range(0, len(ids), config.model.batch_size):
            batch = ids[start : start + config.model.batch_size]
            histories = [table.history(int(i), model.max_length) for i in batch]
            inputs = pad_sequences(histories, model.max_length, device)
            scores = (model.user_vectors(inputs) @ catalog.T).cpu().numpy()
            if not np.isfinite(scores).all():
                raise RuntimeError("nonfinite catalog scores")
            for event, values in zip(batch, scores, strict=True):
                event = int(event)
                target = int(table.targets[event]) - 1
                masked = mask_history(values, [i - 1 for i in table.history(event)], target)
                rank = rank_of_target(masked, target)
                yield {
                    **table.rows[event],
                    "rank": rank,
                    "candidate_count": len(table.items),
                    **metrics_from_rank(rank, config.evaluation.cutoffs),
                }


def combination_dir(context, date, seed, arm):
    return context.recommendations_dir / date / f"seed_{seed}" / arm.lower()


def combination_complete(directory, identity, expected_count):
    try:
        complete = read_json(directory / "complete.json")
        if complete.get("schema_version") != SCHEMA or any(
            complete.get("identity", {}).get(key) != value for key, value in identity.items()
        ):
            return False
        if complete.get("event_count") != expected_count:
            return False
        names = {"sasrec.pt", "training.json", "per_event_metrics.jsonl"}
        if not all((directory / name).is_file() and (directory / name).stat().st_size for name in names):
            return False
        training = read_json(directory / "training.json")
        if training.get("schema_version") != SCHEMA or any(
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


def run_rolling(context, *, force=False):
    from validation.steps import validation_config

    require_torch()
    config = validation_config(context)
    cohort = context.require_ready_cohort()
    table = EventTable(iter_jsonl(context.cohort_dir / "events.jsonl"))
    verify_representations(context, cohort)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    completed = skipped = 0
    for split in cohort["plan"]["splits"]:
        date = split["evaluation_date"]
        ids = {phase: phase_ids(table, split, phase) for phase in split["phases"]}
        probabilities = {}
        for phase in ("selection", "refit"):
            counts = np.bincount(table.targets[ids[phase]], minlength=len(table.items) + 1)
            probabilities[phase] = counts / counts.sum()
        raw_refit = table.select(end=split["phases"]["refit"]["end_ms"], eligible=False)
        frequencies = np.bincount(table.targets[raw_refit], minlength=len(table.items) + 1)
        for seed in config.model.seeds:
            for arm, branch in RECOMMENDATION_ARMS.items():
                identity = {
                    "run_id": context.run_id,
                    "evaluation_date": date,
                    "seed": seed,
                    "arm": arm,
                }
                directory = combination_dir(context, date, seed, arm)
                if not force and combination_complete(directory, identity, len(ids["test"])):
                    skipped += 1
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "complete.json").unlink(missing_ok=True)
                print(f"[Rolling] {date} seed={seed} {arm}: selection/refit/test", flush=True)
                started = time.monotonic()
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
                    score = sum(
                        r["NDCG@10"]
                        for r in evaluate(model, table, ids["validation"], config, device)
                    ) / len(ids["validation"])
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
                refit = []
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
                        "refit_item_frequency": {
                            item: int(frequencies[i + 1]) for i, item in enumerate(table.items)
                        },
                        "elapsed_seconds": time.monotonic() - started,
                        "device": str(device),
                        "environment": {
                            "python": sys.version,
                            "torch": str(torch.__version__),
                            "numpy": np.__version__,
                            "cuda": torch.version.cuda,
                        },
                    },
                )
                write_json(
                    directory / "complete.json",
                    {
                        "schema_version": SCHEMA,
                        "identity": identity,
                        "event_count": count,
                    },
                )
                completed += 1
                del model, optimizer, features
    print(f"[Rolling] completed={completed} skipped={skipped}", flush=True)
    return {"stage": "run-recommendation", "completed": completed, "skipped": skipped}
