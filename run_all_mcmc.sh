#!/usr/bin/env bash
set -euo pipefail

OIDS_FILE="sed_sample.txt"
LOGDIR="mcmc_logs"
NJOBS=2          # number of OIDs to run in parallel
NCORES=8         # cores per MCMC run

mkdir -p "$LOGDIR"

echo "Starting MCMC batch run"
echo "OIDs file : $OIDS_FILE"
echo "Parallel jobs : $NJOBS"
echo "Cores per job : $NCORES"
echo "Logs in : $LOGDIR"
echo

parallel -j "$NJOBS" --bar \
  "echo '>>> Starting {} at \$(date)' && \
   python -m lltypeiip.inference.run_sed_mcmc {} \
     --mode data \
     --nsteps 1500 \
     --burnin 300 \
     --nwalkers 32 \
     --ncores $NCORES \
     --mp fork \
     > ${LOGDIR}/{}.log 2>&1 && \
   echo '<<< Finished {} at \$(date)'" \
  :::: "$OIDS_FILE"

echo
echo "All jobs finished at $(date)"