#!/usr/bin/env bash
set -euo pipefail
release_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $# -ne 1 ]]; then
  echo "Usage: scripts/apply_stresstest_patch.sh STRESSTEST_CHECKOUT" >&2
  exit 2
fi
checkout=$1
revision=$(git -C "$checkout" rev-parse HEAD)
if [[ $revision != e72a8c0* ]]; then
  echo "StressTest checkout must be at upstream revision e72a8c0 (found $revision)." >&2
  exit 2
fi
patch=$release_root/integrations/stresstest/tinystress.patch
git -C "$checkout" apply --check "$patch"
git -C "$checkout" apply "$patch"
echo "Applied TinyStress integration to $checkout"
