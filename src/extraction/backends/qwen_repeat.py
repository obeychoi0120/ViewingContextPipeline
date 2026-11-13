"""Stop generation instead of repeating an earlier output line, independent of engine imports.

A small model can loop by writing the same Graph lines again until the token limit. When the
line being written is token-for-token an earlier complete line, and a line break would come
next, the stop token is raised just above every line-break token. Greedy decoding then ends the
output there instead of starting the repetition, and the backend drops that last duplicate line.
A line that continues past an earlier line (``a - cutting - door; knife`` after
``a - cutting - door``) is not affected.
"""

from __future__ import annotations

REPEAT_STOP_KEY = "stop_on_repeated_line"
# Lines shorter than this many tokens (``none``, a lone bracket) may legitimately repeat.
MIN_LINE_TOKENS = 2
_HASH_SEED = 20261113
_POSITION_TABLE = 1024


def validate_stop_token(params) -> int | None:
    value = (params.extra_args or {}).get(REPEAT_STOP_KEY)
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{REPEAT_STOP_KEY} must be a non-negative stop token id")
    return value


def newline_token_ids(tokenizer) -> list[int]:
    """Token ids whose text contains a line break."""
    return [
        index for index in range(len(tokenizer))
        if "\n" in tokenizer.decode([index], skip_special_tokens=False)
    ]


def drop_repeated_last_line(text: str) -> str:
    """Remove a final line that only repeats an earlier line (left by the stop)."""
    lines = text.rstrip().split("\n")
    if len(lines) > 1 and lines[-1].strip() and lines[-1].strip() in {
        line.strip() for line in lines[:-1]
    }:
        return "\n".join(lines[:-1])
    return text


class LineHashes:
    """Vectorized test: is the line being written identical to an earlier complete line?"""

    def __init__(self, newline_ids, device):
        import torch

        self.newline_ids = sorted(set(newline_ids))
        self.device = device
        self.token_values = None
        self.newline_mask = None
        self.positions = torch.randint(
            -(2**62), 2**62, (_POSITION_TABLE,),
            generator=torch.Generator().manual_seed(_HASH_SEED), dtype=torch.int64,
        ).to(device)

    def _tables(self, width):
        import torch

        if self.newline_mask is None or self.newline_mask.numel() < width:
            mask = torch.zeros(width, dtype=torch.bool)
            ids = [index for index in self.newline_ids if index < width]
            mask[ids] = True
            values = torch.randint(
                -(2**62), 2**62, (width,),
                generator=torch.Generator().manual_seed(_HASH_SEED + width), dtype=torch.int64,
            )
            self.newline_mask, self.token_values = mask.to(self.device), values.to(self.device)
        return self.newline_mask, self.token_values

    def repeated(self, history, valid, vocab_width):
        """history/valid: [rows, width] generated tokens. Returns a [rows] bool tensor."""
        import torch

        rows, width = history.shape
        if width == 0:
            return torch.zeros(rows, dtype=torch.bool, device=history.device)
        newline_mask, token_values = self._tables(vocab_width)
        tokens = history.long().clamp(0, vocab_width - 1)
        newline = newline_mask[tokens] & valid
        content = valid & ~newline
        line = torch.cumsum(newline.long(), dim=1) - newline.long()
        current = newline.long().sum(dim=1)
        columns = torch.arange(width, device=history.device).expand(rows, -1)
        last_break = torch.cummax(torch.where(newline, columns, -1), dim=1).values
        offset = (columns - last_break - 1).clamp(min=0) % _POSITION_TABLE
        contribution = torch.where(content, token_values[tokens] * self.positions[offset], 0)
        hashes = torch.zeros(rows, width + 1, dtype=torch.int64, device=history.device)
        lengths = torch.zeros_like(hashes)
        hashes.scatter_add_(1, line, contribution)
        lengths.scatter_add_(1, line, content.long())
        current_hash = hashes.gather(1, current[:, None])
        current_length = lengths.gather(1, current[:, None])
        earlier = torch.arange(width + 1, device=history.device)[None, :] < current[:, None]
        same = earlier & (hashes == current_hash) & (lengths == current_length)
        return same.any(dim=1) & (current_length[:, 0] >= MIN_LINE_TOKENS)

    def raise_stop(self, logits, rows, stop_ids):
        """For selected rows, put each stop token just above every line-break token."""
        import torch

        newline_mask, _ = self._tables(logits.shape[1])
        breaks = logits.masked_fill(~newline_mask[: logits.shape[1]][None, :], float("-inf"))
        target = torch.nextafter(breaks.max(dim=1).values, torch.tensor(float("inf"), device=logits.device))
        stops = stop_ids.long().clamp(min=0)[:, None]
        current = logits.gather(1, stops)[:, 0]
        raised = torch.where(rows, torch.maximum(current, target.to(logits.dtype)), current)
        logits.scatter_(1, stops, raised[:, None])
        return logits


def _load_newline_ids(vllm_config):
    from transformers import AutoTokenizer

    config = vllm_config.model_config
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer, local_files_only=True)
    return newline_token_ids(tokenizer)


class RepeatedLineStop:
    """vLLM V1 batch state; history comes from each request's live output list."""

    def __init__(self, vllm_config, device, is_pin_memory):
        self.vllm_config = vllm_config
        self.device = device
        self.requests = {}
        self.hashes = None

    @classmethod
    def validate_params(cls, params):
        validate_stop_token(params)

    def is_argmax_invariant(self):
        return False

    def _ensure_hashes(self):
        if self.hashes is None:
            self.hashes = LineHashes(_load_newline_ids(self.vllm_config), self.device)

    def update_state(self, batch_update):
        if batch_update is None:
            return
        for index in batch_update.removed:
            self.requests.pop(index, None)
        for index, params, _prompt_ids, output_ids in batch_update.added:
            stop = validate_stop_token(params)
            self.requests.pop(index, None)
            if stop is not None:
                self._ensure_hashes()
                # Keep the engine's live output list, not a copy of its current contents.
                self.requests[index] = (stop, output_ids)
        for source, target, direction in batch_update.moved:
            source_state = self.requests.pop(source, None)
            target_state = self.requests.pop(target, None)
            if source_state is not None:
                self.requests[target] = source_state
            if direction.name == "SWAP" and target_state is not None:
                self.requests[source] = target_state

    def apply(self, logits):
        import torch

        if not self.requests:
            return logits
        rows = len(logits)
        width = max(len(output) for _, output in self.requests.values())
        history = torch.zeros(rows, width, dtype=torch.long)
        valid = torch.zeros(rows, width, dtype=torch.bool)
        stops = torch.full((rows,), -1, dtype=torch.long)
        for row, (stop, output) in self.requests.items():
            if output:
                history[row, : len(output)] = torch.tensor(output, dtype=torch.long)
                valid[row, : len(output)] = True
            stops[row] = stop
        history, valid, stops = (t.to(logits.device) for t in (history, valid, stops))
        repeated = self.hashes.repeated(history, valid, logits.shape[1]) & (stops >= 0)
        return self.hashes.raise_stop(logits, repeated, stops)


class RepeatedLineStopV2:
    """vLLM V2 request slots; history is read from the on-device token buffers."""

    def __init__(self, vllm_config, req_states):
        import torch

        self.vllm_config = vllm_config
        self.req_states = req_states
        self.stops = [-1] * req_states.max_num_reqs
        self.stops_gpu = torch.full(
            (req_states.max_num_reqs,), -1, device=req_states.device, dtype=torch.long
        )
        self.dirty = False
        self.hashes = None

    def add_request(self, req_idx, params):
        stop = validate_stop_token(params)
        # Slots are reused after completion/preemption.
        self.stops[req_idx] = -1 if stop is None else stop
        self.dirty = True
        if stop is not None and self.hashes is None:
            self.hashes = LineHashes(_load_newline_ids(self.vllm_config), self.req_states.device)
        return stop is not None

    def apply_staged_writes(self):
        if self.dirty:
            import torch

            # Snapshot rather than mutate a tensor still used by an earlier step.
            self.stops_gpu = torch.tensor(self.stops, device=self.req_states.device, dtype=torch.long)
            self.dirty = False

    def apply(self, logits, ctx):
        import torch

        if self.hashes is None or not any(self.stops[int(slot)] >= 0 for slot in ctx.idx_mapping_np):
            return logits
        states = self.req_states
        slots = ctx.expanded_idx_mapping.long()
        starts = states.prompt_len.gpu[slots].long()
        positions_now = ctx.pos.long()
        num_reqs = len(ctx.idx_mapping_np)
        upper_bounds = ctx.seq_lens_upper_bound_np[:num_reqs] - states.prompt_len.np[ctx.idx_mapping_np]
        width = max(0, int(upper_bounds.max(initial=0)))
        rows = len(logits)
        columns = torch.arange(width, device=logits.device)
        positions = starts[:, None] + columns
        committed_end = torch.minimum(states.total_len.gpu[slots].long(), positions_now + 1)
        buffer = states.all_token_ids.gpu
        history = buffer[slots[:, None], positions.clamp_max(buffer.shape[1] - 1)].long()
        history = torch.where(positions < committed_end[:, None], history, 0)
        # This step's input/draft tokens through each row's own position (sentinel column = width).
        draft = (
            (slots[:, None] == slots[None, :])
            & (positions_now[None, :] >= starts[:, None])
            & (positions_now[None, :] <= positions_now[:, None])
        )
        target = torch.where(draft, positions_now[None, :] - starts[:, None], width)
        extended = torch.cat([history, torch.zeros(rows, 1, dtype=torch.long, device=logits.device)], 1)
        extended.scatter_(1, target.clamp(0, width), ctx.input_ids.long()[None, :].expand(rows, -1))
        history = extended[:, :width]
        valid = positions <= positions_now[:, None]
        stops = self.stops_gpu[slots]
        repeated = self.hashes.repeated(history, valid, logits.shape[1]) & (stops >= 0)
        return self.hashes.raise_stop(logits, repeated, stops)
