from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .config import ValidationConfig
from .io import atomic_write_jsonl, read_jsonl
from .metrics import metrics_from_rank
from .model import SASRec, catalog_score_batches, in_batch_loss, require_torch, save_checkpoint, seed_everything, torch
from .scoring import mask_history, rank_of_target


RECOMMENDATION_ARMS: dict[str, str | None] = {
    "SASRec_ID": None,
    "SASRec_GRAPH_QWEN": "graph_qwen",
    "SASRec_GRAPH_GEMINI": "graph_gemini",
    "SASRec_DESC": "desc",
}


def _internal(items: list[str], item_index: dict[str, int]) -> list[int]:
    return [item_index[item] + 1 for item in items]


def _metric(scores: np.ndarray, history: list[int], target: int, cutoffs: list[int]) -> dict[str, float]:
    return metrics_from_rank(rank_of_target(mask_history(scores, history, target)), cutoffs)


def _validation_ndcg(model, histories_internal, histories, targets, config, device) -> float:
    values: list[float] = []
    for start, batch_scores in catalog_score_batches(model, histories_internal, batch_size=config.model.batch_size, device=device):
        for offset, scores in enumerate(batch_scores):
            values.append(_metric(scores, histories[start + offset], targets[start + offset], [10])["NDCG@10"])
    return float(np.mean(values))


def train_recommendation_arms(config: ValidationConfig, runtime: dict[str, Any]) -> dict[str, Any]:
    require_torch()
    root = Path(runtime["run_root"])
    representations_dir = Path(runtime["paths"]["representations_dir"])
    sequences = read_jsonl(root / "data" / "cohort" / "sequences.jsonl")
    import json
    item_index = json.loads((representations_dir / "item_index.json").read_text(encoding="utf-8"))
    index_item = {index: item_id for item_id, index in item_index.items()}
    catalog = read_jsonl(root / "data" / "cohort" / "catalog.jsonl")
    item_content = {row["item_id"]: row["content_id"] for row in catalog}
    branch_features = {
        branch: np.load(representations_dir / f"{branch}_embeddings.npz")["values"]
        for branch in ("graph_qwen", "graph_gemini", "desc")
    }
    arms = dict(RECOMMENDATION_ARMS)
    train_sequences = [_internal(row["train"], item_index) for row in sequences]
    valid_internal = train_sequences
    test_internal = [items + [item_index[row["valid_target"]] + 1] for items, row in zip(train_sequences, sequences)]
    valid_history = [[item - 1 for item in items[-10:]] for items in valid_internal]
    test_history = [[item - 1 for item in items[-10:]] for items in test_internal]
    valid_targets = [item_index[row["valid_target"]] for row in sequences]
    test_targets = [item_index[row["test_target"]] for row in sequences]
    frequency = Counter(item - 1 for items in train_sequences for item in items)
    nonzero = sorted(frequency.values())
    median = nonzero[len(nonzero) // 2] if nonzero else 0
    buckets = ["cold" if frequency[target] == 0 else "low" if frequency[target] <= median else "warm" for target in test_targets]
    output = Path(runtime["paths"]["recommendations_dir"])
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total = len(config.model.seeds) * len(arms)
    completed = 0
    started = perf_counter()
    for seed in config.model.seeds:
        for arm, branch in arms.items():
            print(f"[PHASE] run_recommendation train seed={seed} arm={arm}", flush=True)
            seed_everything(seed)
            features = None if branch is None else branch_features[branch]
            arm_kind = (
                "id"
                if branch is None
                else "graph"
                if branch.startswith("graph_")
                else "desc"
            )
            model = SASRec(len(item_index), config.model.max_sequence_length, config.model.embedding_dim, config.model.num_blocks, config.model.num_heads, config.model.dropout, arm=arm_kind, item_features=features).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.learning_rate, weight_decay=0.1)
            best_ndcg = -1.0
            best_state = None
            stale = 0
            rng = np.random.default_rng(seed)
            history: list[dict[str, Any]] = []
            for epoch in range(1, config.model.max_epochs + 1):
                model.train()
                losses: list[float] = []
                order = rng.permutation(len(train_sequences))
                for start in range(0, len(order), config.model.batch_size):
                    batch = [train_sequences[index] for index in order[start:start + config.model.batch_size] if len(train_sequences[index]) >= 2]
                    if not batch:
                        continue
                    optimizer.zero_grad()
                    loss = in_batch_loss(model, batch, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                ndcg = _validation_ndcg(model, valid_internal, valid_history, valid_targets, config, device)
                history.append({"epoch": epoch, "loss": float(np.mean(losses)), "NDCG@10": ndcg})
                if ndcg > best_ndcg:
                    best_ndcg, stale = ndcg, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= config.model.patience:
                        break
            if best_state is None:
                raise RuntimeError(f"training produced no checkpoint for {arm} seed {seed}")
            model.load_state_dict(best_state)
            for start, batch_scores in catalog_score_batches(model, test_internal, batch_size=config.model.batch_size, device=device):
                for offset, scores in enumerate(batch_scores):
                    index = start + offset
                    masked = mask_history(scores, test_history[index], test_targets[index])
                    rank = rank_of_target(masked, test_targets[index])
                    count = min(20, len(masked))
                    top = np.argpartition(-masked, count - 1)[:count]
                    top = top[np.argsort(-masked[top], kind="stable")].tolist()
                    top_item_ids = [index_item[value] for value in top]
                    rows.append({
                        "seed": seed, "user_id": sequences[index]["user_id"], "arm": arm, "branch": branch,
                        "history_stratum": sequences[index]["stratum"], "target_frequency_bucket": buckets[index], "rank": rank,
                        "target_item_id": index_item[test_targets[index]], "target_content_id": item_content[index_item[test_targets[index]]],
                        "top_item_ids": top_item_ids, "top_content_ids": [item_content[item] for item in top_item_ids],
                        **metrics_from_rank(rank, config.evaluation.cutoffs),
                    })
            checkpoint = output / "checkpoints" / f"seed_{seed}" / arm.lower() / "sasrec.pt"
            save_checkpoint(checkpoint, model, {"seed": seed, "arm": arm, "branch": branch, "best_ndcg_at_10": best_ndcg})
            runs.append({"seed": seed, "arm": arm, "branch": branch, "best_ndcg_at_10": best_ndcg, "epochs": history, "checkpoint": str(checkpoint)})
            completed += 1
            elapsed = perf_counter() - started
            eta = elapsed / completed * (total - completed)
            print(f"[PROGRESS] run_recommendation {completed}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s arm={arm} seed={seed}", flush=True)
    metrics_path = output / "per_user_metrics.jsonl"
    atomic_write_jsonl(metrics_path, rows)
    return {
        "run_id": runtime["run_id"],
        "user_count": len(sequences),
        "per_user_metrics": str(metrics_path),
        "runs": runs,
    }
