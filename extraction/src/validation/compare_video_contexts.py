from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.common.manifest import CANONICAL_MANIFEST_PATH
from src.common.output_paths import custom_output_root
from src.validation.video_context_comparison import (
    PER_CONTENT_FILENAME,
    REPORT_FILENAME,
    run_video_context_comparison,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare complete Graph and Reference video context sets."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=CANONICAL_MANIFEST_PATH,
        help="Input video manifest CSV. Defaults to contracts/manifest.csv.",
    )
    parser.add_argument(
        "--context-dir",
        dest="context_dir",
        type=Path,
        required=True,
        help="Directory containing {content_id}_context_graph_ond.json files.",
    )
    parser.add_argument(
        "--context-ref-dir",
        dest="context_ref_dir",
        type=Path,
        required=True,
        help="Directory containing {content_id}_context_graph_ref.json files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Report directory. Defaults to OUTPUT_SAVE_PATH/reports/viewing_context/comparisons/{mode}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load_dotenv(PROJECT_ROOT / "config" / ".env")
    try:
        mode = args.context_dir.parent.name
        if mode not in {"fixed_15s", "fixed_30s", "shot_wise"}:
            raise ValueError(
                "context directory must be nested under fixed_15s, fixed_30s, or shot_wise"
            )
        report_dir = args.report_dir or (
            _required_output_root()
            / "reports"
            / "viewing_context"
            / "comparisons"
            / mode
        )
        report = run_video_context_comparison(
            manifest_path=args.manifest,
            context_dir=args.context_dir,
            context_ref_dir=args.context_ref_dir,
            report_dir=report_dir,
        )
    except (OSError, ValueError) as exc:
        print(f"Comparison error: {exc}", file=sys.stderr)
        return 1

    print(
        f"Compared {report['paired_content_count']} VC_graph/VC_graph_ref pairs"
    )
    print(f"Wrote {report_dir / REPORT_FILENAME}")
    print(f"Wrote {report_dir / PER_CONTENT_FILENAME}")
    return 0


def _required_output_root() -> Path:
    value = os.getenv("OUTPUT_SAVE_PATH", "").strip()
    if not value:
        raise ValueError(
            "OUTPUT_SAVE_PATH is required in config/.env or the environment"
        )
    return custom_output_root(value)


if __name__ == "__main__":
    raise SystemExit(main())
