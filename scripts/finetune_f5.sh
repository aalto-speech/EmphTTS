#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -lt 1 ]]; then
  echo "Usage: scripts/finetune_f5.sh CHECKPOINT [Hydra overrides...]" >&2
  exit 2
fi
checkpoint=$1
shift
exec python -m emphtts.tts.train.train --config-name EmphTTS "ckpts.pretrain=$checkpoint" "$@"
