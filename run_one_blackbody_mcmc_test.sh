#!/usr/bin/env bash
set -u
set -o pipefail

oid="$1"
logdir="$2"
ncores="$3"

# ---- cache (shared across both thickness runs) ----
cachedir="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/dusty_npz_cache"

# ---- shell thickness values ----
THICK_VALUES=(2.0 5.0)

# ---- MCMC parameters (TEST) ----
NSTEPS=1500
BURNIN=400

mkdir -p "$logdir" "$cachedir"

# Loop over both thickness values
for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  GRID_CSV="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/blackbody_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_blackbody_thick_${thick_str}.csv"

  # sanity check for this thickness's grid
  [ -f "$GRID_CSV" ] || { echo "!!! Missing blackbody grid CSV: $GRID_CSV"; exit 4; }

  workdir="/tmp/lltypeiip_dusty_work_blackbody/${oid}_thick${thick_str}"
  mkdir -p "$workdir"

  logfile="$logdir/${oid}_thick${thick_str}.log"

  {
    echo ">>> Starting BLACKBODY-MCMC TEST for $oid thick=$thick at $(date)"
    echo "TEST MODE: nsteps=$NSTEPS, burnin=$BURNIN"
    echo "workdir=$workdir"
    echo "cachedir=$cachedir"
    echo "grid_csv=$GRID_CSV"
    echo "shell_thickness=$thick"
    echo

    # Make output line-buffered so you see progress in the log immediately
    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --grid-csv "$GRID_CSV" \
      --shell-thickness "$thick" \
      --mode mixture \
      --nsteps "$NSTEPS" \
      --burnin "$BURNIN" \
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
      echo "<<< Finished BLACKBODY-MCMC TEST $oid thick=$thick at $(date)"
    else
      echo
      echo "!!! FAILED BLACKBODY-MCMC TEST $oid thick=$thick exit=$ec at $(date)"
      # DON'T exit here - continue to next thickness or let the script finish naturally
    fi
  } > "$logfile" 2>&1
  
  # Only exit if this thickness failed AND we want to stop on failure
  # For now, we'll continue even on failure to try both thicknesses
  # If you want to stop on first failure, uncomment the next 3 lines:
  # if [ "$ec" -ne 0 ]; then
  #   exit "$ec"
  # fi

done  # ← This closes the loop properly

echo ">>> Completed both thickness TEST runs for $oid at $(date)"