from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_public_clis_import_without_optional_backend_modules() -> None:
    script = r'''
import importlib.abc
import sys

blocked = ("google.genai", "faster_whisper", "moviepy", "paddleocr")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
import extraction
import extraction.cli
import validation.cli
import viewing_context_pipeline.cli

loaded = set(sys.modules)
assert not any(name in loaded for name in blocked)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    [
                        str(Path(__file__).resolve().parents[2] / "src"),
                        os.environ.get("PYTHONPATH", ""),
                    ],
                )
            ),
        },
    )
    assert completed.returncode == 0, completed.stderr
