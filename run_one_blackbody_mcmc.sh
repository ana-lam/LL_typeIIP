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

# ---- sanity check all grids ----
for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  fitted_csv="${PROJECT_ROOT}/fitted_grids/blackbody/thick_${thick_str}/all_objects_blackbody_thick${thick_str}_fitted.csv"
  [ -f "$fitted_csv" ] || { echo "!!! Missing fitted grid CSV: $fitted_csv"; exit 4; }
  grid_csv="${PROJECT_ROOT}/dusty_runs/blackbody_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_blackbody_thick_${thick_str}.csv"
  [ -f "$grid_csv" ] || { echo "!!! Missing template grid CSV: $grid_csv"; exit 4; }
done


# ---- function to run one thickness ----
run_thick() {
  local thick="$1"
  local thick_str="${thick//./_}"
  local fitted_csv="${PROJECT_ROOT}/fitted_grids/blackbody/thick_${thick_str}/all_objects_blackbody_thick${thick_str}_fitted.csv"
  local GRID_CSV="${PROJECT_ROOT}/dusty_runs/blackbody_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_blackbody_thick_${thick_str}.csv"
  local workdir="/tmp/lltypeiip_dusty_work_blackbody/${oid}_thick${thick_str}"
  local logfile="$logdir/${oid}_thick${thick_str}.log"

  [ -f "$GRID_CSV" ] || { echo "!!! Missing blackbody grid CSV: $GRID_CSV"; exit 4; }
  mkdir -p "$workdir"





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
    echo "sed_pkl=$sed_pkl"
    echo "grid_csv=$GRID_CSV"
    echo "fitted_grid_csv=$fitted_csv"
    echo "shell_thickness=$thick"
    echo

    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --fitted-grid-csv "$fitted_csv" \
      --grid-csv "$grid_csv" \
      --shell-thickness "$thick" \
      --mode mixture \
      --nsteps "$NSTEPS" \
      --burnin "$BURNIN" \
      --nwalkers 64 \
      --ncores "$ncores" \
      --mp fork \
      --progress-every 500 \
      --workdir "$workdir" \
      --cache-dir "$cachedir" \
      --cache-ndigits 4 \
      --cache-max 100000 \
      --seed "$SEED"

    local ec=$?
    if [ "$ec" -eq 0 ]; then
      echo
      echo "<<< Finished BLACKBODY-MCMC PRODUCTION $oid thick=$thick at $(date)"
    else
      echo
      echo "!!! FAILED BLACKBODY-MCMC PRODUCTION $oid thick=$thick exit=$ec at $(date)"
    fi
    exit "$ec"
  } > "$logfile" 2>&1
}

# ---- run both thicknesses in parallel ----
pids=()
for thick in "${THICK_VALUES[@]}"; do
  run_thick "$thick" &
  pids+=($!)
done


overall=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "!!! Thickness ${THICK_VALUES[$i]} FAILED for $oid" >&2
    overall=1
  fi
done

if [ "$overall" -eq 0 ]; then
  echo ">>> Completed both thickness PRODUCTION runs for $oid at $(date)"
else
  echo "!!! One or more thickness runs FAILED for $oid" >&2
  exit 1
fi