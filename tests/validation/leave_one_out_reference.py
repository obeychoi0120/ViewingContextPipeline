"""Scalar loss/ranking oracle for the current event-sequence semantics."""

import numpy as np
from validation.metrics import metrics_from_rank
from validation.model import pad_sequences, torch
from validation.scoring import mask_history, rank_of_target

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
