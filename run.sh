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

PIPELINE="${PIPELINE:-config/pipelines/microlens_graph_vs_desc_pilot.yaml}"
LOCAL_CONFIG="${LOCAL_CONFIG:-config/local.yaml}"
STAGE="${STAGE:-prepare_data}"

ARGS=(
  "$STAGE"
  --config "$PIPELINE"
  --local-config "$LOCAL_CONFIG"
)

ARGS+=(--run-id "$RUN_ID")

exec python -m viewing_context_pipeline "${ARGS[@]}" "$@"
