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
TEMPLATE_GRID_CSV="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/template_grids/grid_summary_${TEMPLATE_TAG}.csv"

# ---- cache (shared across both thickness runs) ----
cachedir="/home/cal/analam/Documents/LL_typeIIP/dusty_runs/dusty_npz_cache"

# ---- shell thickness values ----
THICK_VALUES=(2.0 5.0)

sed_pkl="${TAIL_SED_DIR}/${oid}_tail_sed.pkl"

# sanity checks (do once before loop)
[ -f "$sed_pkl" ] || { echo "!!! Missing SED pickle: $sed_pkl"; exit 2; }
[ -f "$TEMPLATE_PATH" ] || { echo "!!! Missing template: $TEMPLATE_PATH"; exit 3; }
[ -f "$TEMPLATE_GRID_CSV" ] || { echo "!!! Missing template grid CSV: $TEMPLATE_GRID_CSV"; exit 4; }

mkdir -p "$logdir" "$cachedir"

# Loop over both thickness values
for thick in "${THICK_VALUES[@]}"; do
  workdir="/tmp/lltypeiip_dusty_work_template/${oid}_thick${thick}"
  mkdir -p "$workdir"
  
  thick_str="${thick//./_}"  # 2.0 -> 2_0
  logfile="$logdir/${oid}_thick${thick_str}.log"

  {
    echo ">>> Starting TEMPLATE-MCMC $oid thick=$thick at $(date)"
    echo "workdir=$workdir"
    echo "cachedir=$cachedir"
    echo "sed_pkl=$sed_pkl"
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
      echo "<<< Finished TEMPLATE-MCMC $oid thick=$thick at $(date)"
    else
      echo
      echo "!!! FAILED TEMPLATE-MCMC $oid thick=$thick exit=$ec at $(date)"
      exit "$ec"
    fi
  } > "$logfile" 2>&1
done

echo ">>> Completed both thickness runs for $oid at $(date)"