#!/usr/bin/env bash
set -euo pipefail

OIDS_FILE="sed_sample.txt"
LOGDIR="mcmc_logs"
NJOBS=4         # number of OIDs to run in parallel
NCORES=4         # cores per MCMC run

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
     --nsteps 1200 \
     --burnin 300 \
     --nwalkers 24 \
     --ncores $NCORES \
     --mp fork \
     --progress-every 100 \
     --workdir /tmp/lltypeiip_dusty_work \
     --cache-dir /tmp/lltypeiip_dusty_cache \
     --cache-ndigits 4 \
     --cache-max 5000 \
     > ${LOGDIR}/{}.log 2>&1 && \
   echo '<<< Finished {} at \$(date)'" \
  :::: "$OIDS_FILE"

echo
echo "All jobs finished at $(date)"