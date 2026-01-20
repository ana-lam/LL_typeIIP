#!/usr/bin/env bash
set -u  # keep -u, but drop -e so we can log failures
set -o pipefail

oid="$1"
logdir="$2"
ncores="$3"

workdir="/tmp/lltypeiip_dusty_work/$oid"
cachedir="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/cache"

mkdir -p "$workdir" "$logdir" "$cachedir"

logfile="$logdir/$oid.log"

{
  echo ">>> Starting $oid at $(date)"
  echo "workdir=$workdir"
  echo "cachedir=$cachedir"
  echo

  # Make output line-buffered so you see progress in the log immediately
  stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
    --mode mixture \
    --nsteps 6000 \
    --burnin 2000 \
    --nwalkers 32 \
    --ncores "$ncores" \
    --mp fork \
    --progress-every 250 \
    --workdir "$workdir" \
    --cache-dir "$cachedir" \
    --cache-ndigits 4 \
    --cache-max 5000

  ec=$?
  if [ "$ec" -eq 0 ]; then
    echo
    echo "<<< Finished $oid at $(date)"
  else
    echo
    echo "!!! FAILED $oid exit=$ec at $(date)"
  fi
  exit "$ec"
} > "$logfile" 2>&1
