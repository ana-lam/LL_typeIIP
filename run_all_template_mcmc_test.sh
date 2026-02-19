#!/usr/bin/env bash
set -euo pipefail

OIDS_FILE="sed_sample.txt"
LOGDIR="mcmc_logs_template"
NJOBS=2
NCORES=4

mkdir -p "$LOGDIR"

echo "Starting TEMPLATE-MCMC batch run"
echo "OIDs file      : $OIDS_FILE"
echo "Parallel jobs  : $NJOBS"
echo "Cores per job  : $NCORES"
echo "Logs in        : $LOGDIR"
echo

parallel -j "$NJOBS" --eta --joblog "${LOGDIR}/joblog.tsv" --lb \
  ./run_one_template_mcmc_test.sh {} "$LOGDIR" "$NCORES" \
  :::: "$OIDS_FILE"

echo
echo "All TEMPLATE-MCMC jobs finished at $(date)"
