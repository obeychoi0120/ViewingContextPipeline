"""Generated-token-only penalty under V1 batches and V2 request-slot layouts."""

from abc import ABC, abstractmethod
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from extraction.backends.qwen_penalty import (
    GeneratedTokenPenalty, GeneratedTokenPenaltyV2, PENALTY_KEY,
)

DEVICES = ['cpu', 'cuda']


def params(penalty):
    return SimpleNamespace(extra_args={PENALTY_KEY: penalty})


@pytest.fixture(params=DEVICES)
def device(request):
    torch = pytest.importorskip('torch')
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    return request.param


def buffers(device, histories, prompts, *, totals=None, prefill=None):
    import torch

    total = [len(row) for row in histories] if totals is None else totals
    width = max(len(row) for row in histories) + 3
    values = [row + [0] * (width - len(row)) for row in histories]
    return SimpleNamespace(
        device=device, max_num_reqs=len(values), vocab_size=12,
        all_token_ids=SimpleNamespace(gpu=torch.tensor(values, device=device)),
        prompt_len=SimpleNamespace(np=np.array(prompts), gpu=torch.tensor(prompts, device=device)),
        prefill_len=SimpleNamespace(gpu=torch.tensor(prefill or total, device=device)),
        total_len=SimpleNamespace(gpu=torch.tensor(total, device=device)),
    )


def context(device, slots, positions, input_ids, upper_bounds):
    import torch

    unique = list(dict.fromkeys(slots))
    return SimpleNamespace(
        expanded_idx_mapping=torch.tensor(slots, device=device),
        idx_mapping=torch.tensor(unique, device=device), idx_mapping_np=np.array(unique),
        pos=torch.tensor(positions, device=device), input_ids=torch.tensor(input_ids, device=device),
        seq_lens_upper_bound_np=np.array(upper_bounds),
    )


def scores(device, rows):
    import torch

    logits = torch.tensor([0., 4., -6., 8., -10., 12., -14., 16., -18., 20., -float('inf'), 24.], device=device)
    return logits.repeat(rows, 1)


def reference(logits, histories, prompt_lengths, penalties, slots, positions, inputs, totals=None):
    """Independent Python oracle: one sign-dependent scale per observed token."""
    expected = logits.clone()
    for row, (slot, position) in enumerate(zip(slots, positions, strict=True)):
        end = min(position + 1, (totals or [len(h) for h in histories])[slot])
        generated = set(histories[slot][prompt_lengths[slot]:end])
        for other, (other_slot, other_position) in enumerate(zip(slots, positions, strict=True)):
            if other_slot == slot and prompt_lengths[slot] <= other_position <= position:
                generated.add(inputs[other])
        for token in generated:
            value = expected[row, token]
            expected[row, token] = value * penalties[slot] if value < 0 else value / penalties[slot]
    return expected


def test_v1_live_outputs_moves_and_prompt_exclusion(device):
    import torch

    penalty = GeneratedTokenPenalty(None, device, False)
    outputs = [2, 2, 3, 0]
    penalty.update_state(SimpleNamespace(removed=[], added=[(0, params(2), [1, 4], outputs)], moved=[]))
    logits = scores(device, 2)
    expected = logits.clone()
    expected[0, 2] *= 2
    expected[0, 3] /= 2
    torch.testing.assert_close(penalty.apply(logits), expected)
    outputs.append(4)
    penalty.update_state(SimpleNamespace(removed=[], added=[], moved=[(0, 1, SimpleNamespace(name='UNIDIRECTIONAL'))]))
    expected = scores(device, 2)
    expected[1, 2] *= 2
    expected[1, 3] /= 2
    expected[1, 4] *= 2
    torch.testing.assert_close(penalty.apply(scores(device, 2)), expected)
    penalty.update_state(SimpleNamespace(removed=[1], added=[], moved=[]))
    torch.testing.assert_close(penalty.apply(scores(device, 2)), scores(device, 2))


def test_v2_permuted_slots_resumption_and_unique_tokens(device):
    import torch

    histories = [[1, 4, 2, 2, 3, 0], [2, 5], [1, 6, 7, 6, 10]]
    prompts = [2, 1, 1]
    state = buffers(device, histories, prompts, prefill=[6, 2, 5])
    penalty = GeneratedTokenPenaltyV2(None, state)
    penalties = [2., 1., 1.5]
    for slot, factor in enumerate(penalties):
        assert penalty.add_request(slot, params(factor)) == (factor > 1)
    penalty.apply_staged_writes()
    slots, positions, inputs = [2, 0, 1], [4, 5, 1], [10, 0, 5]
    ctx = context(device, slots, positions, inputs, [5, 6, 2])
    expected = reference(scores(device, 3), histories, prompts, penalties, slots, positions, inputs)
    snapshots = {name: getattr(state, name).gpu.clone() for name in ('all_token_ids', 'prompt_len', 'prefill_len', 'total_len')}
    torch.testing.assert_close(penalty.apply(scores(device, 3), ctx), expected)
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(getattr(state, name).gpu, snapshot)
    # Reads live GPU history on the next step; does not freeze the initial output.
    state.all_token_ids.gpu[0, 6] = 4
    state.total_len.gpu[0] = 7
    histories[0].append(4)
    ctx = context(device, [0], [6], [4], [7])
    expected = reference(scores(device, 1), histories, prompts, penalties, [0], [6], [4])
    torch.testing.assert_close(penalty.apply(scores(device, 1), ctx), expected)


def test_v2_draft_rows_no_future_or_cross_request_leak(device):
    import torch

    histories, prompts = [[1, 4], [2, 6, 5]], [2, 1]
    state = buffers(device, histories, prompts)
    penalty = GeneratedTokenPenaltyV2(None, state)
    penalty.add_request(0, params(2))
    penalty.add_request(1, params(1.5))
    penalty.apply_staged_writes()
    # First draft row still ends at the last prompt token, so has no generated history.
    # Later rows may see their own earlier drafts but not future drafts or other slots.
    slots, positions, inputs = [0, 1, 0, 0], [1, 2, 2, 3], [4, 5, 7, 2]
    ctx = context(device, slots, positions, inputs, [4, 3])
    expected = reference(scores(device, 4), histories, prompts, [2, 1.5], slots, positions, inputs)
    torch.testing.assert_close(penalty.apply(scores(device, 4), ctx), expected)


def test_v2_recycled_slot_clears_old_penalty_and_snapshot(device):
    import torch

    state = buffers(device, [[1, 2, 3]], [1])
    penalty = GeneratedTokenPenaltyV2(None, state)
    penalty.add_request(0, params(2))
    penalty.apply_staged_writes()
    previous = penalty.penalties_gpu
    assert penalty.add_request(0, params(1)) is False
    penalty.apply_staged_writes()
    assert previous is not penalty.penalties_gpu
    torch.testing.assert_close(previous, torch.tensor([2.], device=device))
    ctx = context(device, [0], [2], [3], [3])
    original = scores(device, 1)
    assert penalty.apply(original, ctx) is original
    torch.testing.assert_close(original, scores(device, 1))
    # Even if the new request uses a penalty, prompt-only prefill must see no old output.
    penalty.add_request(0, params(1.5))
    penalty.apply_staged_writes()
    state.prompt_len.np[0] = 3
    state.prompt_len.gpu[0] = 3
    torch.testing.assert_close(penalty.apply(scores(device, 1), ctx), scores(device, 1))


@pytest.mark.parametrize('value', [True, '1.1', 0, 0.5, float('nan'), float('inf')])
def test_both_interfaces_reject_invalid_penalty(value):
    with pytest.raises(ValueError, match='finite and at least 1'):
        GeneratedTokenPenalty.validate_params(params(value))
    penalty = GeneratedTokenPenaltyV2(None, buffers('cpu', [[1]], [1]))
    with pytest.raises(ValueError, match='finite and at least 1'):
        penalty.add_request(0, params(value))


@pytest.mark.parametrize('with_v2', [False, True])
def test_fqcn_accepts_old_and_new_loader_contracts(monkeypatch, with_v2):
    class V1(ABC):
        @abstractmethod
        def __init__(self, config, device, pin): ...
        @abstractmethod
        def apply(self, logits): ...
        @abstractmethod
        def update_state(self, update): ...
        @abstractmethod
        def is_argmax_invariant(self): ...
    class V2(ABC):
        @abstractmethod
        def apply(self, logits, context): ...
    old = ModuleType('vllm.v1.sample.logits_processor')
    old.LogitsProcessor = V1
    monkeypatch.setitem(sys.modules, old.__name__, old)
    for name in ('vllm', 'vllm.v1', 'vllm.v1.worker', 'vllm.v1.worker.gpu',
                 'vllm.v1.worker.gpu.sample', 'vllm.v1.worker.gpu.sample.logits_processor'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    if with_v2:
        new = ModuleType('vllm.v1.worker.gpu.sample.logits_processor.interface')
        new.LogitsProcessor = V2
        monkeypatch.setitem(sys.modules, new.__name__, new)
    spec = importlib.util.spec_from_file_location('compat_probe', Path('src/extraction/backends/qwen_logits.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.GeneratedTokenLogitsProcessor
    assert issubclass(cls, V1)
    legacy = cls(None, 'cpu', False)
    assert legacy.is_argmax_invariant() is False
    legacy.update_state(SimpleNamespace(removed=[], added=[], moved=[]))
    if with_v2:
        assert issubclass(cls, V2)
        modern = cls(None, buffers('cpu', [[1, 2]], [1]))
        assert modern.add_request(0, params(1.1)) is True
        modern.apply_staged_writes()
        modern.apply(scores('cpu', 1), context('cpu', [0], [1], [2], [2]))
