from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cohort import prepare_cohort
from .config import load_config
from .experiment import train_and_evaluate
from .features import materialize_representations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate incremental Viewing Context ranking value on MicroLens.")
    parser.add_argument("--config", default="config/pilot_1k.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="Validate configuration and report input availability.")
    subparsers.add_parser("prepare-cohort", help="Build the deterministic user cohort and cohort-union catalog.")
    subparsers.add_parser("materialize-representations", help="Encode paired Graph and Description profiles with local BGE.")
    subparsers.add_parser("run-experiment", help="Train independent ID, Graph, and Description SASRec arms.")
    subparsers.add_parser("run-all", help="Run cohort, representation, training, and reporting stages.")
    return parser


def preflight(config_path: str | Path) -> dict[str, object]:
    config = load_config(config_path)
    paths = {
        "pairs_tsv": config.dataset.pairs_tsv,
        "videos_dir": config.dataset.videos_dir,
        "vp_graph_dir": config.dataset.vp_graph_dir,
        "vp_desc_dir": config.dataset.vp_desc_dir,
        "encoder_model_path": config.encoder.model_path,
    }
    try:
        import torch  # noqa: F401
        torch_available = True
    except ModuleNotFoundError:
        torch_available = False
    return {
        "config_valid": True,
        "run_id": config.run_id,
        "inputs": {name: {"path": str(path), "exists": path.exists()} for name, path in paths.items()},
        "torch_available": torch_available,
        "ready_for_cohort": all(paths[name].exists() for name in ("pairs_tsv", "videos_dir")),
        "ready_for_materialization": all(paths[name].exists() for name in ("vp_graph_dir", "vp_desc_dir", "encoder_model_path")),
        "ready_for_training": torch_available,
    }


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "preflight":
        result = preflight(args.config)
    else:
        config = load_config(args.config)
        if args.command == "prepare-cohort":
            result = prepare_cohort(config)
        elif args.command == "materialize-representations":
            result = materialize_representations(config)
        elif args.command == "run-experiment":
            result = train_and_evaluate(config)
        else:
            prepare_cohort(config)
            materialize_representations(config)
            result = train_and_evaluate(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
