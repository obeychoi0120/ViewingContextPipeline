from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import yaml


def test_public_clis_import_without_optional_backend_modules() -> None:
    script = r'''
import importlib.abc
import sys

blocked = ("google.genai", "torch", "transformers")

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise ModuleNotFoundError(fullname)
        return None

sys.meta_path.insert(0, Blocker())
import extraction
import extraction.cli
import validation.cli
import pipeline_runtime

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


def test_plan_only_cli_runs_without_optional_imports_assets_or_external_processes(tmp_path):
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "config/pipeline.yaml").read_text(encoding="utf-8"))
    config["artifacts_root"] = str(tmp_path / "artifacts")
    config["validation"]["cohort"]["user_count"] = 1
    config["data"] = {
        "pairs_tsv": str(tmp_path / "pairs.tsv"),
        "videos_dir": str(tmp_path / "absent-videos"),
        "titles_csv": str(tmp_path / "absent-titles.csv"),
    }
    (tmp_path / "config").mkdir()
    (tmp_path / "config/pipeline.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (tmp_path / "pairs.tsv").write_text("u1\t1 2 3 4 5\n", encoding="utf-8")
    script = r'''
import importlib.abc
from pathlib import Path
import subprocess
import sys

blocked = ("torch", "transformers", "google.genai")
class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(tuple(name + "." for name in blocked)):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, Blocker())

def forbidden(*args, **kwargs):
    raise AssertionError("plan-only must not launch ffprobe or another process")
subprocess.run = forbidden
from pipeline_runtime import RunContext
from validation.cli import main
load = RunContext.load
RunContext.load = lambda run_id: load(run_id, root=Path(sys.argv[1]))
assert main(["prepare-cohort", "--run-id", "plan", "--plan-only"]) == 0
assert not any(name in sys.modules for name in blocked)
'''
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "artifacts/plan/data/cohort/required_items.jsonl").is_file()
