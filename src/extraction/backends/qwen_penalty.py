from __future__ import annotations

import math


PENALTY_KEY = "generated_token_repetition_penalty"


class GeneratedTokenPenalty:
    """vLLM batch state and vectorized penalty, independent of engine imports."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.requests = {}

    @classmethod
    def validate_params(cls, params):
        penalty = (params.extra_args or {}).get(PENALTY_KEY, 1.0)
        if type(penalty) not in (int, float) or not math.isfinite(penalty) or penalty < 1:
            raise ValueError("generated-token repetition penalty must be finite and at least 1")

    def is_argmax_invariant(self):
        return False

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, _prompt_ids, output_ids in batch_update.added:
            self.validate_params(params)
            penalty = (params.extra_args or {}).get(PENALTY_KEY, 1.0)
            self.requests.pop(index, None)
            if penalty > 1:
                # Keep the engine's live output list, not a copy of its current contents.
                self.requests[index] = (penalty, output_ids)
        for source, target, direction in batch_update.moved:
            source_state = self.requests.pop(source, None)
            target_state = self.requests.pop(target, None)
            if source_state is not None:
                self.requests[target] = source_state
            if direction.name == "SWAP" and target_state is not None:
                self.requests[source] = target_state

    def apply(self, logits):
        import torch

        rows, tokens, penalties = [], [], []
        for row, (penalty, output_ids) in self.requests.items():
            unique = set(output_ids)
            rows.extend([row] * len(unique))
            tokens.extend(unique)
            penalties.extend([penalty] * len(unique))
        if rows:
            row_ids = torch.tensor(rows, device=logits.device, dtype=torch.long)
            token_ids = torch.tensor(tokens, device=logits.device, dtype=torch.long)
            factors = torch.tensor(penalties, device=logits.device, dtype=logits.dtype)
            scores = logits[row_ids, token_ids]
            logits[row_ids, token_ids] = torch.where(scores < 0, scores * factors, scores / factors)
        return logits
