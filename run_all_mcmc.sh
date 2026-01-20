#!/usr/bin/env bash
set -euo pipefail

OIDS_FILE="sed_sample.txt"
LOGDIR="mcmc_logs"
NJOBS=4
NCORES=4

mkdir -p "$LOGDIR"

echo "Starting MCMC batch run"
echo "OIDs file : $OIDS_FILE"
echo "Parallel jobs : $NJOBS"
echo "Cores per job : $NCORES"
echo "Logs in : $LOGDIR"
echo

parallel -j "$NJOBS" --eta --joblog "${LOGDIR}/joblog.tsv" --lb \
  ./run_one_mcmc.sh {} "$LOGDIR" "$NCORES" \
  :::: "$OIDS_FILE"

echo
echo "All jobs finished at $(date)"
