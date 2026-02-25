#!/usr/bin/env bash
set -u
set -o pipefail

oid="$1"
logdir="$2"
ncores="$3"

# ---- inputs ----
TAIL_SED_DIR="/home/cal/analam/Documents/LL_typeIIP/data/tail_seds"
TEMPLATE_PATH="/home/cal/analam/Documents/LL_typeIIP/data/typeiip_spectral_templates/sn2p_flux.v1.2.dat"
TEMPLATE_TAG="nugent_iip"

# ---- cache dir ----
cachedir="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/dusty_npz_cache"

# ---- shell thickness values ----
THICK_VALUES=(2.0 5.0)

# ---- MCMC parameters (PRODUCTION) ----
NSTEPS=8000
BURNIN=2000

sed_pkl="${TAIL_SED_DIR}/${oid}_tail_sed.pkl"

# sanity checks
[ -f "$sed_pkl" ]       || { echo "!!! Missing SED pickle: $sed_pkl";   exit 2; }
[ -f "$TEMPLATE_PATH" ] || { echo "!!! Missing template: $TEMPLATE_PATH"; exit 3; }

mkdir -p "$logdir" "$cachedir"

# ---- sanity check all grids ----
for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  grid_csv="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/template_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_${TEMPLATE_TAG}_thick_${thick_str}.csv"
  [ -f "$grid_csv" ] || { echo "!!! Missing template grid CSV: $grid_csv"; exit 4; }
done

# ---- function to run one thickness ----
run_thick() {
  local thick="$1"
  local thick_str="${thick//./_}"
  local grid_csv="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/template_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_${TEMPLATE_TAG}_thick_${thick_str}.csv"
  local workdir="/tmp/lltypeiip_dusty_work_template/${oid}_thick${thick_str}"
  local logfile="${logdir}/${oid}_thick${thick_str}.log"

  mkdir -p "$workdir"

  {
    echo ">>> Starting TEMPLATE-MCMC PRODUCTION for $oid thick=$thick at $(date)"
    echo "PRODUCTION MODE: nsteps=$NSTEPS, burnin=$BURNIN"
    echo "workdir=$workdir"
    echo "cachedir=$cachedir"
    echo "sed_pkl=$sed_pkl"
    echo "grid-csv=$grid_csv"
    echo "shell_thickness=$thick"
    echo

    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --sed-pkl "$sed_pkl" \
      --template-path "$TEMPLATE_PATH" \
      --template-tag "$TEMPLATE_TAG" \
      --grid-csv "$grid_csv" \
      --tstar-dummy 6000 \
      --shell-thickness "$thick" \
      --mode mixture \
      --nsteps "$NSTEPS" \
      --burnin "$BURNIN" \
      --nwalkers 32 \
      --ncores "$ncores" \
      --mp fork \
      --progress-every 500 \
      --workdir "$workdir" \
      --cache-dir "$cachedir" \
      --cache-ndigits 4 \
      --cache-max 5000

    local ec=$?
    if [ "$ec" -eq 0 ]; then
      echo
      echo "<<< Finished TEMPLATE-MCMC PRODUCTION $oid thick=$thick at $(date)"
    else
      echo
      echo "!!! FAILED TEMPLATE-MCMC PRODUCTION $oid thick=$thick exit=$ec at $(date)"
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