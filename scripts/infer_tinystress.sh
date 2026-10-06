#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec accelerate launch -m emphtts.tts.eval.eval_infer_batch_durpred "$@"
