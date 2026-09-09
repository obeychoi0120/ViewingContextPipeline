from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from tqdm import tqdm

from .config import ValidationConfig
from .io import atomic_write_jsonl, read_jsonl
from .metrics import metrics_from_rank
from .model import (
    SASRec,
    catalog_score_batches,
    in_batch_loss,
    require_torch,
    save_checkpoint,
    seed_everything,
    torch,
)
from .recommendation_contracts import (
    ARCHITECTURE_VERSION,
    RECOMMENDATION_ARMS,
    TRAINING_RUN_SCHEMA_VERSION,
    TRAINING_RUNS_FILENAME,
)
from .scoring import mask_history, rank_of_target, top_k_rows


def _internal(items: list[str], item_index: dict[str, int]) -> list[int]:
    return [item_index[item] + 1 for item in items]


def _metric(
    scores: np.ndarray,
    history: list[int],
    target: int,
    cutoffs: list[int],
) -> dict[str, float]:
    masked = mask_history(scores, history, target)
    return metrics_from_rank(rank_of_target(masked, target), cutoffs)


def _validation_ndcg(
    model: SASRec,
    histories_internal: list[list[int]],
    histories: list[list[int]],
    targets: list[int],
    config: ValidationConfig,
    device: "torch.device",
) -> float:
    values: list[float] = []
    for start, batch_scores in catalog_score_batches(
        model,
        histories_internal,
        batch_size=config.model.batch_size,
        device=device,
    ):
        for offset, scores in enumerate(batch_scores):
            values.append(
                _metric(
                    scores,
                    histories[start + offset],
                    targets[start + offset],
                    [10],
                )["NDCG@10"]
            )
    return float(np.mean(values))


def popularity_probabilities(
    sequences: list[list[int]],
    item_count: int,
    power: float,
) -> np.ndarray:
    counts = np.zeros(item_count + 1, dtype=np.float64)
    for sequence in sequences:
        for item_id in sequence:
            if not 1 <= item_id <= item_count:
                raise ValueError(f"item id outside catalog: {item_id}")
            counts[item_id] += 1
    powered = np.power(counts[1:], power)
    denominator = float(powered.sum())
    if not np.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("cannot build popularity probabilities from empty training data")
    probabilities = np.zeros(item_count + 1, dtype=np.float32)
    probabilities[0] = 1.0
    probabilities[1:] = (powered / denominator).astype(np.float32)
    return probabilities


def _arm_kind(branch: str) -> str:
    if branch == "metadata":
        return "metadata"
    if branch.startswith("graph_"):
        return "graph"
    if branch == "desc":
        return "desc"
    raise ValueError(f"unsupported recommendation branch: {branch}")


def _new_model(
    config: ValidationConfig,
    *,
    item_count: int,
    branch: str,
    features: np.ndarray,
    device: "torch.device",
) -> SASRec:
    return SASRec(
        item_count,
        config.model.max_sequence_length,
        config.model.embedding_dim,
        config.model.num_blocks,
        config.model.num_heads,
        config.model.dropout,
        arm=_arm_kind(branch),
        item_features=features,
    ).to(device)


def _train_epoch(
    model: SASRec,
    optimizer: "torch.optim.Optimizer",
    sequences: list[list[int]],
    order: np.ndarray,
    config: ValidationConfig,
    probabilities: np.ndarray,
    device: "torch.device",
) -> float:
    model.train()
    losses: list[float] = []
    for start in range(0, len(order), config.model.batch_size):
        batch = [
            sequences[index]
            for index in order[start : start + config.model.batch_size]
            if len(sequences[index]) >= 2
        ]
        if not batch:
            continue
        optimizer.zero_grad()
        loss = in_batch_loss(model, batch, device, probabilities)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses or not np.isfinite(losses).all():
        raise RuntimeError("training epoch produced no finite losses")
    return float(np.mean(losses))


def _optimizer(model: SASRec, config: ValidationConfig) -> "torch.optim.Optimizer":
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=0.1,
    )


def _training_run_record(
    *,
    run_id: str,
    seed: int,
    arm: str,
    branch: str,
    selection_history: list[dict[str, Any]],
    best_ndcg: float,
    best_epoch: int,
    refit_history: list[dict[str, Any]],
    checkpoint: Path,
    candidate_count: int,
    elapsed_seconds: float,
    max_epochs: int,
) -> dict[str, Any]:
    return {
        "schema_version": TRAINING_RUN_SCHEMA_VERSION,
        "architecture_version": ARCHITECTURE_VERSION,
        "run_id": run_id,
        "seed": seed,
        "arm": arm,
        "branch": branch,
        "candidate_count": candidate_count,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": elapsed_seconds,
        "selection": {
            "epochs_completed": len(selection_history),
            "early_stopped": len(selection_history) < max_epochs,
            "best_validation": {
                "metric": "NDCG@10",
                "value": best_ndcg,
                "epoch": best_epoch,
            },
            "epochs": selection_history,
        },
        "refit": {
            "data": "train+valid_target",
            "epochs_completed": len(refit_history),
            "epochs": refit_history,
        },
    }


def _persist_training_runs(path: Path, runs: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, runs)


@dataclass(frozen=True)
class _RecommendationInputs:
    sequences: list[dict[str, Any]]
    item_index: dict[str, int]
    index_item: dict[int, str]
    item_content: dict[str, str]
    branch_features: dict[str, np.ndarray]
    train_sequences: list[list[int]]
    refit_sequences: list[list[int]]
    valid_history: list[list[int]]
    test_history: list[list[int]]
    valid_targets: list[int]
    test_targets: list[int]
    buckets: list[str]


def _prepare_inputs(config: ValidationConfig, runtime: dict[str, Any]) -> _RecommendationInputs:
    root = Path(runtime["run_root"])
    representations_dir = Path(runtime["paths"]["representations_dir"])
    sequences = read_jsonl(root / "data" / "cohort" / "sequences.jsonl")
    item_index = json.loads((representations_dir / "item_index.json").read_text(encoding="utf-8"))
    index_item = {index: item_id for item_id, index in item_index.items()}
    catalog = read_jsonl(root / "data" / "cohort" / "catalog.jsonl")
    expected_item_index = {row["item_id"]: index for index, row in enumerate(catalog)}
    if item_index != expected_item_index:
        raise RuntimeError(
            "representation item index does not match the cohort catalog; "
            "rerun embed-representations"
        )
    item_content = {row["item_id"]: row["content_id"] for row in catalog}
    branch_features: dict[str, np.ndarray] = {}
    expected_shape = (len(catalog), config.encoder.embedding_dim)
    for branch in RECOMMENDATION_ARMS.values():
        path = representations_dir / f"{branch}_embeddings.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != {"values"}:
                    raise ValueError("NPZ must contain exactly the values array")
                values = np.asarray(archive["values"], dtype=np.float32)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"failed to load {branch} representations: {path}") from exc
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise RuntimeError(
                f"invalid {branch} representations: expected {expected_shape}, "
                f"observed {values.shape}"
            )
        branch_features[branch] = values

    train_sequences = [_internal(row["train"], item_index) for row in sequences]
    refit_sequences = [
        items + [item_index[row["valid_target"]] + 1]
        for items, row in zip(train_sequences, sequences, strict=True)
    ]
    valid_history = [
        [item - 1 for item in items[-config.model.max_sequence_length :]]
        for items in train_sequences
    ]
    test_history = [
        [item - 1 for item in items[-config.model.max_sequence_length :]]
        for items in refit_sequences
    ]
    valid_targets = [item_index[row["valid_target"]] for row in sequences]
    test_targets = [item_index[row["test_target"]] for row in sequences]
    frequency = Counter(item - 1 for items in train_sequences for item in items)
    nonzero = sorted(frequency.values())
    median = nonzero[len(nonzero) // 2] if nonzero else 0
    buckets = [
        "cold" if frequency[target] == 0 else "low" if frequency[target] <= median else "warm"
        for target in test_targets
    ]

    return _RecommendationInputs(
        sequences=sequences,
        item_index=item_index,
        index_item=index_item,
        item_content=item_content,
        branch_features=branch_features,
        train_sequences=train_sequences,
        refit_sequences=refit_sequences,
        valid_history=valid_history,
        test_history=test_history,
        valid_targets=valid_targets,
        test_targets=test_targets,
        buckets=buckets,
    )


def _select_epoch(config, inputs, seed, arm, branch, features, device, selection_probabilities):
    seed_everything(seed)
    selection_model = _new_model(
        config,
        item_count=len(inputs.item_index),
        branch=branch,
        features=features,
        device=device,
    )
    selection_optimizer = _optimizer(selection_model, config)
    best_ndcg = -1.0
    best_epoch = 0
    stale = 0
    selection_rng = np.random.default_rng(seed)
    selection_history: list[dict[str, Any]] = []
    for epoch in range(1, config.model.max_epochs + 1):
        loss = _train_epoch(
            selection_model,
            selection_optimizer,
            inputs.train_sequences,
            selection_rng.permutation(len(inputs.train_sequences)),
            config,
            selection_probabilities,
            device,
        )
        ndcg = _validation_ndcg(
            selection_model,
            inputs.train_sequences,
            inputs.valid_history,
            inputs.valid_targets,
            config,
            device,
        )
        selection_history.append({"epoch": epoch, "loss": loss, "NDCG@10": ndcg})
        if ndcg > best_ndcg:
            best_ndcg, best_epoch, stale = ndcg, epoch, 0
        else:
            stale += 1
            if stale >= config.model.patience:
                break
    if best_epoch <= 0:
        raise RuntimeError(f"selection produced no best epoch for {arm} seed {seed}")
    del selection_optimizer, selection_model

    return best_epoch, best_ndcg, selection_history


def _refit_model(config, inputs, seed, branch, features, device, refit_probabilities, best_epoch):
    seed_everything(seed)
    model = _new_model(
        config,
        item_count=len(inputs.item_index),
        branch=branch,
        features=features,
        device=device,
    )
    optimizer = _optimizer(model, config)
    refit_rng = np.random.default_rng(seed)
    refit_history: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        loss = _train_epoch(
            model,
            optimizer,
            inputs.refit_sequences,
            refit_rng.permutation(len(inputs.refit_sequences)),
            config,
            refit_probabilities,
            device,
        )
        refit_history.append({"epoch": epoch, "loss": loss})

    return model, optimizer, refit_history


def _evaluate_arm(config, inputs, model, seed, arm, branch, device, rows):
    for start, batch_scores in catalog_score_batches(
        model,
        inputs.refit_sequences,
        batch_size=config.model.batch_size,
        device=device,
    ):
        for offset, scores in enumerate(batch_scores):
            index = start + offset
            masked = mask_history(
                scores,
                inputs.test_history[index],
                inputs.test_targets[index],
            )
            rank = rank_of_target(masked, inputs.test_targets[index])
            count = min(20, len(masked))
            top = top_k_rows(masked, count)
            top_item_ids = [inputs.index_item[value] for value in top]
            rows.append(
                {
                    "seed": seed,
                    "user_id": inputs.sequences[index]["user_id"],
                    "arm": arm,
                    "branch": branch,
                    "candidate_count": len(inputs.item_index),
                    "history_stratum": inputs.sequences[index]["stratum"],
                    "target_frequency_bucket": inputs.buckets[index],
                    "rank": rank,
                    "target_item_id": inputs.index_item[inputs.test_targets[index]],
                    "target_content_id": inputs.item_content[
                        inputs.index_item[inputs.test_targets[index]]
                    ],
                    "top_item_ids": top_item_ids,
                    "top_content_ids": [inputs.item_content[item] for item in top_item_ids],
                    **metrics_from_rank(rank, config.evaluation.cutoffs),
                }
            )


def _save_refit_checkpoint(
    output, inputs, model, seed, arm, branch, best_ndcg, best_epoch, refit_history
):
    checkpoint = output / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
    save_checkpoint(
        checkpoint,
        model,
        {
            "architecture_version": ARCHITECTURE_VERSION,
            "training_phase": "refit",
            "seed": seed,
            "arm": arm,
            "branch": branch,
            "selection_best_ndcg_at_10": best_ndcg,
            "selection_best_epoch": best_epoch,
            "refit_epochs_completed": len(refit_history),
            "candidate_count": len(inputs.item_index),
        },
    )
    return checkpoint


def train_recommendation_arms(
    config: ValidationConfig,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    require_torch()
    inputs = _prepare_inputs(config, runtime)
    output = Path(runtime["paths"]["recommendations_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    training_runs_path = output / TRAINING_RUNS_FILENAME
    _persist_training_runs(training_runs_path, runs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(config.model.seeds) * len(RECOMMENDATION_ARMS)

    selection_probabilities = popularity_probabilities(
        inputs.train_sequences,
        len(inputs.item_index),
        config.model.popularity_power,
    )
    refit_probabilities = popularity_probabilities(
        inputs.refit_sequences,
        len(inputs.item_index),
        config.model.popularity_power,
    )

    with tqdm(total=total, desc="Recommendation", unit="run") as progress:
        for seed in config.model.seeds:
            for arm, branch in RECOMMENDATION_ARMS.items():
                progress.set_postfix(seed=seed, arm=arm)
                print(
                    f"[PHASE] run_recommendation select seed={seed} arm={arm}",
                    flush=True,
                )
                arm_started = perf_counter()
                features = inputs.branch_features[branch]

                best_epoch, best_ndcg, selection_history = _select_epoch(
                    config,
                    inputs,
                    seed,
                    arm,
                    branch,
                    features,
                    device,
                    selection_probabilities,
                )

                print(
                    f"[PHASE] run_recommendation refit seed={seed} arm={arm} epochs={best_epoch}",
                    flush=True,
                )
                model, optimizer, refit_history = _refit_model(
                    config,
                    inputs,
                    seed,
                    branch,
                    features,
                    device,
                    refit_probabilities,
                    best_epoch,
                )

                _evaluate_arm(config, inputs, model, seed, arm, branch, device, rows)

                checkpoint = _save_refit_checkpoint(
                    output,
                    inputs,
                    model,
                    seed,
                    arm,
                    branch,
                    best_ndcg,
                    best_epoch,
                    refit_history,
                )
                runs.append(
                    _training_run_record(
                        run_id=runtime["run_id"],
                        seed=seed,
                        arm=arm,
                        branch=branch,
                        selection_history=selection_history,
                        best_ndcg=best_ndcg,
                        best_epoch=best_epoch,
                        refit_history=refit_history,
                        checkpoint=checkpoint,
                        candidate_count=len(inputs.item_index),
                        elapsed_seconds=perf_counter() - arm_started,
                        max_epochs=config.model.max_epochs,
                    )
                )
                _persist_training_runs(training_runs_path, runs)
                progress.update(1)

    metrics_path = output / "per_user_metrics.jsonl"
    atomic_write_jsonl(metrics_path, rows)
    return {
        "run_id": runtime["run_id"],
        "user_count": len(inputs.sequences),
        "per_user_metrics": str(metrics_path),
        "training_runs": str(training_runs_path),
        "runs": runs,
    }
