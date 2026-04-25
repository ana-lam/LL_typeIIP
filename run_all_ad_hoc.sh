#!/usr/bin/env bash
set -euo pipefail

LOGDIR="mcmc_logs_template"
NCORES=2
SEED=0317

mkdir -p "$LOGDIR"

echo "Starting no_i_band ad hoc MCMC runs at $(date)"

for oid in ZTF21abnlhxs ZTF21acpqqgu ZTF22aaywnyg; do
  echo "Launching $oid"
  ./run_one_ad_hoc.sh "$oid" "$LOGDIR" "$NCORES" "$SEED" &
done

wait
echo "All no_i_band runs finished at $(date)"