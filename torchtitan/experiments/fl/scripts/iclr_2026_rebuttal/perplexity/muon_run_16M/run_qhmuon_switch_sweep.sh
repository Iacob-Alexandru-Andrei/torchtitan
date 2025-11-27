#!/usr/bin/env bash
# Sweep quasi-hyperbolic Muon (DDP) over (v, switch_scale) pairs.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
RUN_SCRIPT="${SCRIPT_DIR}/qhmuon_ddp.sh"

PAIRS=(
  "0.95 1.0"
  "0.96 2.0"
  "0.97 1.4142135623730951"
  "0.98 1.4142135623730951"
  "0.99 1.0"
)

DRY_RUN=${DRY_RUN:-false}
RUN_INDEX=${RUN_INDEX:-}
RUN_INDEX_OFFSET=${RUN_INDEX_OFFSET:-0}

index=0
for pair in "${PAIRS[@]}"; do
  read -r v_value scale_value <<< "${pair}"
  index=$((index + 1))
  if (( index <= RUN_INDEX_OFFSET )); then
    continue
  fi
  if [[ -n "${RUN_INDEX}" && "${index}" -ne "${RUN_INDEX}" ]]; then
    continue
  fi
  echo "Launching v=${v_value}, switch_scale=${scale_value}"
  if [[ "${DRY_RUN}" == "true" ]]; then
    continue
  fi
  QHMUON_V="${v_value}" SWITCH_SCALE="${scale_value}" "${RUN_SCRIPT}" "$@"
done
