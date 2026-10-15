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


class GeneratedTokenPenaltyV2:
    """V2 request slots and GPU token buffers, excluding the original prompt."""

    def __init__(self, vllm_config, req_states):
        import torch

        self.req_states = req_states
        self.penalties = [1.0] * req_states.max_num_reqs
        self.penalties_gpu = torch.ones(
            req_states.max_num_reqs, device=req_states.device, dtype=torch.float32
        )
        self.dirty = False

    def add_request(self, req_idx, params):
        GeneratedTokenPenalty.validate_params(params)
        penalty = (params.extra_args or {}).get(PENALTY_KEY, 1.0)
        # Slots are reused after completion/preemption, including penalty=1.
        self.penalties[req_idx] = penalty
        self.dirty = True
        return penalty > 1.0

    def apply_staged_writes(self):
        if self.dirty:
            import torch

            # Snapshot rather than mutate a tensor still used by an earlier step.
            self.penalties_gpu = torch.tensor(
                self.penalties, device=self.req_states.device, dtype=torch.float32
            )
            self.dirty = False

    def apply(self, logits, ctx):
        import torch

        if not any(self.penalties[int(slot)] > 1 for slot in ctx.idx_mapping_np):
            return logits
        states = self.req_states
        slots = ctx.expanded_idx_mapping.long()
        starts = states.prompt_len.gpu[slots].long()
        ends = torch.minimum(states.total_len.gpu[slots].long(), ctx.pos.long() + 1)
        rows, vocab = logits.shape
        # The extra column is a sentinel: stale/padded history never marks token 0.
        seen = torch.zeros((rows, vocab + 1), device=logits.device, dtype=torch.bool)
        # vLLM may pad lengths for CUDA graphs; mappings contain only active requests.
        num_reqs = len(ctx.idx_mapping_np)
        upper_bounds = (
            ctx.seq_lens_upper_bound_np[:num_reqs]
            - states.prompt_len.np[ctx.idx_mapping_np]
        )
        history_width = max(0, int(upper_bounds.max(initial=0)))
        if history_width:
            positions = starts[:, None] + torch.arange(history_width, device=logits.device)
            valid = positions < ends[:, None]
            history = states.all_token_ids.gpu[
                slots[:, None], positions.clamp_max(states.all_token_ids.gpu.shape[1] - 1)
            ].long()
            seen.scatter_(1, torch.where(valid, history, vocab), True)
        # Include current-step input/draft tokens through this row's position.
        # Rows may be reordered or repeated; future drafts and other slots are excluded.
        draft_valid = (
            (slots[:, None] == slots[None, :])
            & (ctx.pos[None, :] >= starts[:, None])
            & (ctx.pos[None, :] <= ctx.pos[:, None])
        )
        drafts = ctx.input_ids.long()[None, :].expand(rows, -1)
        seen.scatter_(1, torch.where(draft_valid, drafts, vocab), True)
        factors = self.penalties_gpu[slots].to(logits.dtype)[:, None]
        scaled = torch.where(logits < 0, logits * factors, logits / factors)
        logits.copy_(torch.where(seen[:, :vocab], scaled, logits))
        return logits
