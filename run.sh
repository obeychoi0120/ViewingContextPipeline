#!/usr/bin/env bash
set -euo pipefail

if [[ $# -eq 0 || -z "$1" ]]; then
  echo "Usage: $0 RUN_ID [pipeline arguments...]" >&2
  echo "RUN_ID must be an explicit directory name, for example 1k_pilot_260824." >&2
  exit 2
fi
RUN_ID="$1"
shift

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ARGS=(
  run
  --run-id "$RUN_ID"
)

exec python -m viewing_context_pipeline "${ARGS[@]}" "$@"
