#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python -m emphtts.tts.eval.eval_tinystress_testset "$@"
