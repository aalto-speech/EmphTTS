#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/train_grpo_with_whistress.sh WHISTRESS_CHECKOUT [Hydra overrides...]" >&2
  exit 2
fi
checkout=$1
shift
if [[ ! -d "$checkout/whistress" ]]; then
  echo "Expected a local WhiStress checkout containing whistress/" >&2
  exit 2
fi
export PYTHONPATH="$checkout${PYTHONPATH:+:$PYTHONPATH}"
exec python -m emphtts.duration.train_grpo grpo.use_stress_metric=true grpo.asr_reward_type=wer "$@"
