#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -lt 1 ]]; then
  echo "usage: bash run.sh RUN_ID [pipeline options]" >&2
  exit 2
fi

RUN_ID=$1
shift
python -m viewing_context_pipeline run --run-id "$RUN_ID" "$@"
