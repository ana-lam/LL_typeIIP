#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

oid="$1"
logdir="$2"
ncores="$3"
SEED="${4:-99}"

TAIL_SED_DIR="${PROJECT_ROOT}/data/tail_seds"
TEMPLATE_PATH="${PROJECT_ROOT}/data/typeiip_spectral_templates/sn2p_flux.v1.2.dat"
TEMPLATE_TAG="nugent_iip"
cachedir="${PROJECT_ROOT}/dusty_runs/dusty_npz_cache"
THICK_VALUES=(2.0 5.0)
NSTEPS=15000
BURNIN=3000

sed_pkl="${TAIL_SED_DIR}/${oid}_tail_sed_no_i_band.pkl"

[ -f "$sed_pkl" ]       || { echo "!!! Missing SED pickle: $sed_pkl";   exit 2; }
[ -f "$TEMPLATE_PATH" ] || { echo "!!! Missing template: $TEMPLATE_PATH"; exit 3; }

mkdir -p "$logdir" "$cachedir"

for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  grid_csv="${PROJECT_ROOT}dusty_runs/template_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_${TEMPLATE_TAG}_thick_${thick_str}.csv"
  [ -f "$grid_csv" ] || { echo "!!! Missing template grid CSV: $grid_csv"; exit 4; }
done

run_thick() {
  local thick="$1"
  local thick_str="${thick//./_}"
  local fitted_csv="${PROJECT_ROOT}/fitted_grids/template/thick_${thick_str}/${oid}_template_thick${thick_str}_fitted_no_i_band.csv"
  local grid_csv="${PROJECT_ROOT}/dusty_runs/template_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_${TEMPLATE_TAG}_thick_${thick_str}.csv"
  local workdir="/tmp/lltypeiip_dusty_work_template/${oid}_no_i_band_thick${thick_str}"
  local logfile="${logdir}/${oid}_no_i_band_thick${thick_str}.log"

  mkdir -p "$workdir"

  {
    echo ">>> Starting no_i_band TEMPLATE-MCMC for $oid thick=$thick at $(date)"
    echo "sed_pkl=$sed_pkl"

    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --sed-pkl "$sed_pkl" \
      --template-path "$TEMPLATE_PATH" \
      --template-tag "$TEMPLATE_TAG" \
      --fitted-grid-csv "$fitted_csv" \
      --run-id no_i_band \
      --grid-csv "$grid_csv" \
      --tstar-dummy 6000 \
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
    [ "$ec" -eq 0 ] \
      && echo "<<< Finished $oid thick=$thick at $(date)" \
      || echo "!!! FAILED $oid thick=$thick exit=$ec at $(date)"
    exit "$ec"
  } > "$logfile" 2>&1
}

pids=()
for thick in "${THICK_VALUES[@]}"; do
  run_thick "$thick" &
  pids+=($!)
done

overall=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "!!! Thickness ${THICK_VALUES[$i]} FAILED for $oid" >&2; overall=1; }
done

[ "$overall" -eq 0 ] \
  && echo ">>> Completed both thicknesses for $oid at $(date)" \
  || { echo "!!! One or more thickness runs FAILED for $oid" >&2; exit 1; }