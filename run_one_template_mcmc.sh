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

# ---- cache (shared across both thickness runs) ----
cachedir="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/dusty_npz_cache"

# ---- shell thickness values ----
THICK_VALUES=(2.0 5.0)

# ---- MCMC parameters (PRODUCTION) ----
NSTEPS=15000
BURNIN=5000

sed_pkl="${TAIL_SED_DIR}/${oid}_tail_sed.pkl"

# sanity checks (do once before loop)
[ -f "$sed_pkl" ] || { echo "!!! Missing SED pickle: $sed_pkl"; exit 2; }
[ -f "$TEMPLATE_PATH" ] || { echo "!!! Missing template: $TEMPLATE_PATH"; exit 3; }

mkdir -p "$logdir" "$cachedir"

# Loop over both thickness values
for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  TEMPLATE_GRID_CSV="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/template_grids/silicate_tau_0.55um_fixed_thick_${thick_str}/grid_summary_${TEMPLATE_TAG}_thick_${thick_str}.csv"

  # sanity check for this thickness's grid
  [ -f "$TEMPLATE_GRID_CSV" ] || { echo "!!! Missing template grid CSV: $TEMPLATE_GRID_CSV"; exit 4; }
  
  workdir="/tmp/lltypeiip_dusty_work_template/${oid}_thick${thick_str}"
  mkdir -p "$workdir"
  
  logfile="$logdir/${oid}_thick${thick_str}.log"

  {
    echo ">>> Starting TEMPLATE-MCMC PRODUCTION for $oid thick=$thick at $(date)"
    echo "PRODUCTION MODE: nsteps=$NSTEPS, burnin=$BURNIN"
    echo "workdir=$workdir"
    echo "cachedir=$cachedir"
    echo "sed_pkl=$sed_pkl"
    echo "template_grid_csv=$TEMPLATE_GRID_CSV"
    echo "shell_thickness=$thick"
    echo

    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --sed-pkl "$sed_pkl" \
      --template-path "$TEMPLATE_PATH" \
      --template-tag "$TEMPLATE_TAG" \
      --template-grid-csv "$TEMPLATE_GRID_CSV" \
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

    ec=$?
    if [ "$ec" -eq 0 ]; then
      echo
      echo "<<< Finished TEMPLATE-MCMC PRODUCTION $oid thick=$thick at $(date)"
    else
      echo
      echo "!!! FAILED TEMPLATE-MCMC PRODUCTION $oid thick=$thick exit=$ec at $(date)"
    fi
  } > "$logfile" 2>&1

  if [ "$ec" -ne 0 ]; then
    exit "$ec"
  fi

done  

echo ">>> Completed both thickness PRODUCTION runs for $oid at $(date)"