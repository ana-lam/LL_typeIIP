#!/usr/bin/env bash
set -euo pipefail

OIDS_FILE="sed_sample.txt"
LOGDIR="mcmc_logs"

NJOBS=15
NCORES_PER_THICK=4

mkdir -p "$LOGDIR"

echo "Starting BLACKBODY-MCMC batch run"
echo "OIDs file         : $OIDS_FILE"
echo "Parallel OIDs     : $NJOBS"
echo "Cores/thickness   : $NCORES_PER_THICK"
echo "Cores/OID         : $((NCORES_PER_THICK * 2)) (2 thicknesses in parallel)"
echo "Total cores used  : $((NJOBS * NCORES_PER_THICK * 2))"
echo "Logs in           : $LOGDIR"
echo

parallel -j "$NJOBS" --eta --joblog "${LOGDIR}/joblog.tsv" --lb \
  ./run_one_blackbody_mcmc.sh {} "$LOGDIR" "$NCORES_PER_THICK" 0430 \
  :::: "$OIDS_FILE"

echo
echo "All BLACKBODY-MCMC jobs finished at $(date)"