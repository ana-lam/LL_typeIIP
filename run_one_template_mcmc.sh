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
TEMPLATE_GRID_CSV="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/template_grid_summary_${TEMPLATE_TAG}.csv"

# ---- work/cache ----
workdir="/tmp/lltypeiip_dusty_work_template/${oid}"

mkdir -p "$workdir" "$logdir" "$cachedir"

logfile="$logdir/${oid}.log"
sed_pkl="${TAIL_SED_DIR}/${oid}_tail_sed.pkl"

{
  echo ">>> Starting TEMPLATE-MCMC $oid at $(date)"
  echo "workdir=$workdir"
  echo "cachedir=$cachedir"
  echo "sed_pkl=$sed_pkl"
  echo "template_path=$TEMPLATE_PATH"
  echo "template_grid_csv=$TEMPLATE_GRID_CSV"
  echo

  # sanity checks
  [ -f "$sed_pkl" ] || { echo "!!! Missing SED pickle: $sed_pkl"; exit 2; }
  [ -f "$TEMPLATE_PATH" ] || { echo "!!! Missing template: $TEMPLATE_PATH"; exit 3; }
  [ -f "$TEMPLATE_GRID_CSV" ] || { echo "!!! Missing template grid CSV: $TEMPLATE_GRID_CSV"; exit 4; }

  stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
    --sed-pkl "$sed_pkl" \
    --template-path "$TEMPLATE_PATH" \
    --template-tag "$TEMPLATE_TAG" \
    --template-grid-csv "$TEMPLATE_GRID_CSV" \
    --tstar-dummy 6000 \
    --mode mixture \
    --nsteps 10000 \
    --burnin 3000 \
    --nwalkers 32 \
    --ncores "$ncores" \
    --mp fork \
    --progress-every 500 \
    --workdir "$workdir" \
    --cache-dir "$cachedir" \
    --cache-ndigits 4 \
    --cache-max 5000

  ec=$?
  if [ "$ec" -eq 0 ]; then
    echo
    echo "<<< Finished TEMPLATE-MCMC $oid at $(date)"
  else
    echo
    echo "!!! FAILED TEMPLATE-MCMC $oid exit=$ec at $(date)"
  fi
  exit "$ec"
} > "$logfile" 2>&1
