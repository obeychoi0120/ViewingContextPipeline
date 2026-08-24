#!/usr/bin/env bash
set -euo pipefail

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

if [[ -n "${RUN_ID:-}" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi

exec python -m viewing_context_pipeline "${ARGS[@]}" "$@"
