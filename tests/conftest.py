"""Every regression runs offline on CPU and writes only into its temporary tree."""

import builtins
import io
import os
from pathlib import Path
import random
import socket
import subprocess
import tempfile

import numpy as np
import pytest
import torch

from helpers import make_context


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    torch_rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.manual_seed(7)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    temporary = tmp_path / "temp"
    temporary.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def forbidden(*args, **kwargs):
        raise AssertionError("external network or subprocess attempted during offline regression")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)

    def check_write(file):
        if not isinstance(file, int):
            path = Path(file).resolve()
            if not path.is_relative_to(tmp_path.resolve()):
                raise AssertionError(f"write outside the test temporary tree: {path}")

    def wrap_open(original):
        def guarded(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in "wax+"):
                check_write(file)
            return original(file, mode, *args, **kwargs)

        return guarded

    original_os_open = os.open

    def guarded_os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            check_write(path)
        return original_os_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", wrap_open(builtins.open))
    monkeypatch.setattr(io, "open", wrap_open(io.open))
    monkeypatch.setattr(os, "open", guarded_os_open)
    try:
        yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
        torch.set_rng_state(torch_rng)
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@pytest.fixture
def context(tmp_path):
    return make_context(tmp_path / "repo")
