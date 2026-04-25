#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

oid="$1"
logdir="$2"
ncores="$3"

# ---- cache (shared across both thickness runs) ----
cachedir="${PROJECT_ROOT}/dusty_runs/dusty_npz_cache"

# ---- shell thickness values ----
THICK_VALUES=(2.0 5.0)

# ---- MCMC parameters (PRODUCTION) ----
NSTEPS=10000
BURNIN=3000

mkdir -p "$logdir" "$cachedir"

# Loop over both thickness values
for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  fitted_csv="${PROJECT_ROOT}/fitted_grids/blackbody/thick_${thick_str}/all_objects_blackbody_thick${thick_str}_fitted.csv"
  GRID_CSV="${PROJECT_ROOT}/dusty_runs/blackbody_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_blackbody_thick_${thick_str}.csv"

  # sanity check for this thickness's grid
  [ -f "$GRID_CSV" ] || { echo "!!! Missing blackbody grid CSV: $GRID_CSV"; exit 4; }

  workdir="/tmp/lltypeiip_dusty_work_blackbody/${oid}_thick${thick_str}"
  mkdir -p "$workdir"

  logfile="$logdir/${oid}_thick${thick_str}.log"

  {
    echo ">>> Starting BLACKBODY-MCMC PRODUCTION for $oid thick=$thick at $(date)"
    echo "PRODUCTION MODE: nsteps=$NSTEPS, burnin=$BURNIN"
    echo "workdir=$workdir"
    echo "cachedir=$cachedir"
    echo "grid_csv=$GRID_CSV"
    echo "fitted_grid_csv=$fitted_csv"
    echo "shell_thickness=$thick"
    echo

    # Make output line-buffered so you see progress in the log immediately
    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --grid-csv "$GRID_CSV" \
      --fitted-grid-csv "$fitted_csv" \
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
      echo "<<< Finished BLACKBODY-MCMC PRODUCTION $oid thick=$thick at $(date)"
    else
      echo
      echo "!!! FAILED BLACKBODY-MCMC PRODUCTION $oid thick=$thick exit=$ec at $(date)"
    fi
  } > "$logfile" 2>&1

  if [ "$ec" -ne 0 ]; then
    exit "$ec"
  fi

done  

echo ">>> Completed both thickness PRODUCTION runs for $oid at $(date)"