#!/usr/bin/env bash
set -euo pipefail

LOGDIR="mcmc_logs_template"

NCORES_PER_THICK=2
SEED=0317

mkdir -p "$LOGDIR"

echo "Starting no_i_band ad hoc TEMPLATE-MCMC runs at $(date)"

pids=()
oids=(ZTF21abnlhxs ZTF21acpqqgu ZTF22aaywnyg)

for oid in "${oids[@]}"; do
  echo "Launching $oid"
  ./run_one_ad_hoc.sh "$oid" "$LOGDIR" "$NCORES_PER_THICK" "$SEED" &
  pids+=($!)
done

overall=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!!! ${oids[$i]} FAILED" >&2
    overall=1
  fi
done

[ "$overall" -eq 0 ] && echo "All no_i_band runs finished at $(date)" || exit 1