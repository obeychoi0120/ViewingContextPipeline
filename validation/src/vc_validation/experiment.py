from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .config import ExperimentConfig
from .io import atomic_write_json, atomic_write_jsonl, fingerprint, read_jsonl
from .metrics import metrics_from_rank, paired_bootstrap_ci, paired_relative_bootstrap_ci
from .model import SASRec, catalog_score_batches, in_batch_loss, require_torch, save_checkpoint, seed_everything, torch
from .scoring import mask_history, rank_of_target


ARMS = ["SASRec_ID", "SASRec_GRAPH", "SASRec_DESC"]
ARM_KINDS = {"SASRec_ID": "id", "SASRec_GRAPH": "graph", "SASRec_DESC": "desc"}


def _load_inputs(config: ExperimentConfig) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, np.ndarray]]:
    sequences = read_jsonl(config.output_dir / "cohort" / "sequences.jsonl")
    root = config.output_dir / "representations"
    item_index = json.loads((root / "item_index.json").read_text(encoding="utf-8"))
    features = {
        "graph": np.load(root / "vp_graph_embeddings.npz")["values"],
        "desc": np.load(root / "vp_desc_embeddings.npz")["values"],
    }
    return sequences, item_index, features


def _internal(items: list[str], item_index: dict[str, int]) -> list[int]:
    return [item_index[item] + 1 for item in items]


def _ranking_metrics(scores: np.ndarray, history: list[int], target: int, cutoffs: list[int]) -> dict[str, float]:
    return metrics_from_rank(rank_of_target(mask_history(scores, history, target)), cutoffs)


def _stream_ndcg(model: SASRec, histories_internal: list[list[int]], histories: list[list[int]], targets: list[int], batch_size: int, cutoff: int, device) -> float:
    total = 0.0
    count = 0
    for start, batch_scores in catalog_score_batches(model, histories_internal, batch_size=batch_size, device=device):
        for offset, scores in enumerate(batch_scores):
            total += _ranking_metrics(scores, histories[start + offset], targets[start + offset], [cutoff])[f"NDCG@{cutoff}"]
            count += 1
    return total / count


def _evaluate(model: SASRec, arm: str, histories_internal: list[list[int]], histories: list[list[int]], targets: list[int], config: ExperimentConfig, user_rows: list[dict[str, Any]], target_buckets: list[str], seed: int, device) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start, batch_scores in catalog_score_batches(model, histories_internal, batch_size=config.model.batch_size, device=device):
        for offset, scores in enumerate(batch_scores):
            index = start + offset
            masked = mask_history(scores, histories[index], targets[index])
            rank = rank_of_target(masked, targets[index])
            top_count = min(20, len(masked))
            top_items = np.argpartition(-masked, top_count - 1)[:top_count]
            top_items = top_items[np.argsort(-masked[top_items], kind="stable")].tolist()
            rows.append({
                "seed": seed, "user_id": user_rows[index]["user_id"], "arm": arm,
                "history_stratum": user_rows[index]["stratum"],
                "target_frequency_bucket": target_buckets[index], "rank": rank,
                "top_items": top_items, **metrics_from_rank(rank, config.evaluation.cutoffs),
            })
    return rows


def _new_model(config: ExperimentConfig, item_count: int, arm: str, features: dict[str, np.ndarray], device):
    kind = ARM_KINDS[arm]
    item_features = None if kind == "id" else features[kind]
    return SASRec(item_count, config.model.max_sequence_length, config.model.embedding_dim, config.model.num_blocks, config.model.num_heads, config.model.dropout, arm=kind, item_features=item_features).to(device)


def train_and_evaluate(config: ExperimentConfig) -> dict[str, Any]:
    require_torch()
    sequences, item_index, features = _load_inputs(config)
    train_sequences = [_internal(row["train"], item_index) for row in sequences]
    valid_histories_internal = train_sequences
    test_histories_internal = [sequence + [item_index[row["valid_target"]] + 1] for sequence, row in zip(train_sequences, sequences)]
    valid_histories = [[item - 1 for item in sequence[-10:]] for sequence in valid_histories_internal]
    test_histories = [[item - 1 for item in sequence[-10:]] for sequence in test_histories_internal]
    valid_targets = [item_index[row["valid_target"]] for row in sequences]
    test_targets = [item_index[row["test_target"]] for row in sequences]
    train_frequency = Counter(item - 1 for sequence in train_sequences for item in sequence)
    positive_counts = sorted(train_frequency.values())
    median_count = positive_counts[len(positive_counts) // 2] if positive_counts else 0
    test_buckets = ["cold" if train_frequency[target] == 0 else "low" if train_frequency[target] <= median_count else "warm" for target in test_targets]
    output = config.output_dir / "experiment"
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    run_manifests: list[dict[str, Any]] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in config.model.seeds:
        for arm in ARMS:
            seed_everything(seed)
            model = _new_model(config, len(item_index), arm, features, device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.model.learning_rate, weight_decay=0.1)
            best_ndcg = -1.0
            best_state: dict[str, Any] | None = None
            stale = 0
            rng = np.random.default_rng(seed)
            validation_history: list[dict[str, Any]] = []
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
                ndcg = _stream_ndcg(model, valid_histories_internal, valid_histories, valid_targets, config.model.batch_size, config.evaluation.primary_cutoff, device)
                validation_history.append({"epoch": epoch, "loss": float(np.mean(losses)), "ndcg_at_10": ndcg})
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
            all_rows.extend(_evaluate(model, arm, test_histories_internal, test_histories, test_targets, config, sequences, test_buckets, seed, device))
            checkpoint = output / f"seed_{seed}" / arm.lower() / "sasrec.pt"
            save_checkpoint(checkpoint, model, {"seed": seed, "arm": arm, "best_ndcg_at_10": best_ndcg})
            manifest = {"seed": seed, "arm": arm, "best_ndcg_at_10": best_ndcg, "epochs": validation_history, "checkpoint": str(checkpoint)}
            atomic_write_json(checkpoint.parent / "manifest.json", manifest)
            run_manifests.append(manifest)
    atomic_write_jsonl(output / "per_user_metrics.jsonl", all_rows)
    atomic_write_json(output / "arm_manifest.json", {"schema_version": "independent-sasrec-arms/v1", "arms": ARMS, "shared_checkpoint_within_seed": False, "frozen_content_representations": True, "runs": run_manifests})
    return build_report(config, all_rows, run_manifests)


def _per_user_seed_mean(rows_by_key: dict[tuple[int, str, str], dict[str, Any]], users: list[str], seeds: list[int], arm: str, metric: str) -> np.ndarray:
    return np.asarray([np.mean([rows_by_key[(seed, user, arm)][metric] for seed in seeds]) for user in users], dtype=np.float64)


def build_report(config: ExperimentConfig, rows: list[dict[str, Any]], run_manifests: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["seed"], row["user_id"], row["arm"]): row for row in rows}
    users = sorted({row["user_id"] for row in rows})
    metrics = [f"{name}@{cutoff}" for cutoff in config.evaluation.cutoffs for name in ("HR", "NDCG")]
    summaries = {arm: {metric: float(np.mean([row[metric] for row in rows if row["arm"] == arm])) for metric in metrics} for arm in ARMS}
    graph = _per_user_seed_mean(by_key, users, config.model.seeds, "SASRec_GRAPH", "NDCG@10")
    desc = _per_user_seed_mean(by_key, users, config.model.seeds, "SASRec_DESC", "NDCG@10")
    primary = paired_relative_bootstrap_ci(graph, desc, samples=config.evaluation.bootstrap_samples)
    primary["margin"] = -config.evaluation.non_inferiority_margin
    primary["non_inferior"] = primary["ci_low"] > primary["margin"]
    secondary: dict[str, Any] = {}
    for arm in ("SASRec_GRAPH", "SASRec_DESC"):
        treatment = _per_user_seed_mean(by_key, users, config.model.seeds, arm, "NDCG@10")
        baseline = _per_user_seed_mean(by_key, users, config.model.seeds, "SASRec_ID", "NDCG@10")
        secondary[f"{arm}-SASRec_ID"] = paired_bootstrap_ci(treatment - baseline, samples=config.evaluation.bootstrap_samples)
    diagnostics: dict[str, Any] = {"catalog": {}}
    catalog_size = len(json.loads((config.output_dir / "representations" / "item_index.json").read_text(encoding="utf-8")))
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        recommended = [item for row in arm_rows for item in row["top_items"]]
        top_one = Counter(row["top_items"][0] for row in arm_rows)
        diagnostics["catalog"][arm] = {"top20_coverage": len(set(recommended)) / catalog_size, "top1_concentration": max(top_one.values()) / len(arm_rows)}
    report = {
        "schema_version": "graph-description-noninferiority-report/v1", "run_id": config.run_id,
        "claim_scope": "Graph versus Description next-item ranking non-inferiority",
        "result": "GRAPH_NON_INFERIOR_TO_DESC" if primary["non_inferior"] else "NON_INFERIORITY_NOT_ESTABLISHED",
        "metrics": summaries, "primary_non_inferiority": primary,
        "secondary_id_comparisons": secondary, "diagnostics": diagnostics,
    }
    output = config.output_dir / "experiment"
    atomic_write_json(output / "report.json", report)
    representation_manifest = json.loads((config.output_dir / "representations" / "representation_manifest.json").read_text(encoding="utf-8"))
    checkpoints_complete = len(run_manifests) == len(ARMS) * len(config.model.seeds) and all(Path(row["checkpoint"]).is_file() for row in run_manifests)
    expected_rows = config.cohort.user_count * len(config.model.seeds) * len(ARMS)
    ready = {
        "report_ready": len(users) == config.cohort.user_count and len(rows) == expected_rows and len({row["arm"] for row in rows}) == len(ARMS) and representation_manifest.get("graph_completeness") == 1.0 and representation_manifest.get("desc_completeness") == 1.0 and representation_manifest.get("failure_count") == 0 and checkpoints_complete,
        "user_count": len(users), "seed_count": len({row["seed"] for row in run_manifests}),
        "arm_count": len({row["arm"] for row in rows}), "graph_completeness": representation_manifest.get("graph_completeness"),
        "desc_completeness": representation_manifest.get("desc_completeness"), "checkpoints_complete": checkpoints_complete,
        "artifact_fingerprint": fingerprint({"report": report, "runs": run_manifests}),
    }
    atomic_write_json(output / "report_ready.json", ready)
    return report
