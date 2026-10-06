#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/run_stresstest.sh STRESSTEST_CHECKOUT [StressTest options...]" >&2
  exit 2
fi
checkout=$1
shift
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
exec python -m stresstest.evaluation.main --task ssd --dataset_type tinystress "$@"
