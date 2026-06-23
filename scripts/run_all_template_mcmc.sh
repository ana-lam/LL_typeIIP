#!/usr/bin/env bash
set -euo pipefail

OIDS_FILE="sed_sample.txt"
LOGDIR="mcmc_logs_template"

NJOBS=4
NCORES_PER_THICK=2

mkdir -p "$LOGDIR"

echo "Starting TEMPLATE-MCMC batch run"
echo "OIDs file         : $OIDS_FILE"
echo "Parallel OIDs     : $NJOBS"
echo "Cores/thickness   : $NCORES_PER_THICK"
echo "Cores/OID         : $((NCORES_PER_THICK * 2)) (2 thicknesses in parallel)"
echo "Total cores used  : $((NJOBS * NCORES_PER_THICK * 2))"
echo "Logs in           : $LOGDIR"
echo

taskset -c 0-15 parallel -j "$NJOBS" --eta --joblog "${LOGDIR}/joblog.tsv" --lb \
  ./run_one_template_mcmc.sh {} "$LOGDIR" "$NCORES_PER_THICK" 0426 \
  :::: "$OIDS_FILE"

echo
echo "All TEMPLATE-MCMC jobs finished at $(date)"