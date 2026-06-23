#!/usr/bin/env bash
set -u
set -o pipefail

PROJECT_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

oid="$1"
logdir="$2"
ncores="$3"

# --- seed ---
SEED="${4:-99}"

# ---- inputs ----
TAIL_SED_DIR="${PROJECT_ROOT}/data/tail_seds"
TEMPLATE_PATH="${PROJECT_ROOT}/data/typeiip_spectral_templates/sn2p_flux.v1.2.dat"
TEMPLATE_TAG="nugent_iip"

# ---- cache dir ----
cachedir="${PROJECT_ROOT}/dusty_runs/dusty_npz_cache_template"
BB_MCMC_DIR="${PROJECT_ROOT}/mcmc_results/${oid}"
BB_GRID_DIR="${PROJECT_ROOT}/fitted_grids/blackbody"

# ---- shell thickness values ----
THICK_VALUES=(2.0 5.0)

# ---- MCMC parameters (PRODUCTION) ----
NSTEPS=150000
BURNIN=10000

sed_pkl="${TAIL_SED_DIR}/${oid}_tail_sed.pkl"

# sanity checks
[ -f "$sed_pkl" ]       || { echo "!!! Missing SED pickle: $sed_pkl";   exit 2; }
[ -f "$TEMPLATE_PATH" ] || { echo "!!! Missing template: $TEMPLATE_PATH"; exit 3; }

mkdir -p "$logdir" "$cachedir"

# ---- sanity check all grids ----
for thick in "${THICK_VALUES[@]}"; do
  thick_str="${thick//./_}"
  fitted_csv="${PROJECT_ROOT}/fitted_grids/template/thick_${thick_str}/all_objects_template_thick${thick_str}_fitted.csv"
  [ -f "$fitted_csv" ] || { echo "!!! Missing fitted grid CSV: $fitted_csv"; exit 4; }
  grid_csv="${PROJECT_ROOT}/dusty_runs/template_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_nugent_iip_thick_${thick_str}.csv"
  [ -f "$grid_csv" ] || { echo "!!! Missing template grid CSV: $grid_csv"; exit 4; }
  emulator_path="${PROJECT_ROOT}/dusty_runs/dusty_tmpl_emulator_thick${thick_str}.npz"
  [ -f "$emulator_path" ] || { echo "!!! Missing template emulator: $emulator_path"; exit 5; }
done

make_init_csv() {
  local oid="$1"
  local thick_str="$2"
  local tmpl_fitted_csv="$3"
  local out_csv="$4"

  local bb_fitted_csv="${BB_GRID_DIR}/thick_${thick_str}/all_objects_blackbody_thick${thick_str}_fitted.csv"

  if [ ! -f "$bb_fitted_csv" ]; then
    echo "  [init] No bb grid CSV found at $bb_fitted_csv — using template grid init"
    cp "$tmpl_fitted_csv" "$out_csv"
    return 0
  fi

  python3 - <<PYEOF
import pandas as pd
import numpy as np
import sys

oid          = "${oid}"
bb_csv       = "${bb_fitted_csv}"
tmpl_csv     = "${tmpl_fitted_csv}"
out_csv      = "${out_csv}"

# Load bb grid best for this object
bb_df   = pd.read_csv(bb_csv)
bb_obj  = bb_df[bb_df["oid"] == oid].sort_values("chi2_red") if "oid" in bb_df.columns else bb_df.sort_values("chi2_red")

if bb_obj.empty:
    print(f"  [init] No bb grid rows for {oid} — using template grid init")
    import shutil; shutil.copy(tmpl_csv, out_csv)
    sys.exit(0)

bb_best = bb_obj.iloc[0]
print(f"  [init] bb grid best: Td={bb_best['tdust']:.0f}K  tau={bb_best['tau']:.4f}  chi2={bb_best['chi2_red']:.2f}")

# Load template grid
tmpl_df = pd.read_csv(tmpl_csv)

# Create init row from bb best — inject into template grid as row 0
init_row = tmpl_df.iloc[0].copy()
init_row["tdust"]    = float(bb_best["tdust"])
init_row["tau"]      = float(bb_best["tau"])
init_row["scale"]    = float(bb_best["scale"]) if "scale" in bb_best else float(tmpl_df.iloc[0]["scale"])
init_row["chi2_red"] = -1.0   # sentinel: marks as bb-init row

out_df = pd.concat([init_row.to_frame().T, tmpl_df], ignore_index=True)
out_df.to_csv(out_csv, index=False)
print(f"  [init] Saved init CSV: {out_csv}  ({len(out_df)} rows)")
PYEOF
}

# ---- function to run one thickness ----
run_thick() {
  local thick="$1"
  local thick_str="${thick//./_}"
  local fitted_csv="${PROJECT_ROOT}/fitted_grids/template/thick_${thick_str}/all_objects_template_thick${thick_str}_fitted.csv"
  local grid_csv="${PROJECT_ROOT}/dusty_runs/template_grids/silicate_tau_0.55um_thick_${thick_str}/grid_summary_nugent_iip_thick_${thick_str}.csv"
  local emulator_path="${PROJECT_ROOT}/dusty_runs/dusty_tmpl_emulator_thick${thick_str}.npz"
  local workdir="${PROJECT_ROOT}/dusty_runs/mcmc_workdir/${oid}_thick${thick_str}_smpl"
  local logfile="${logdir}/${oid}_thick${thick_str}_smpl.log"

  mkdir -p "$workdir"

  local init_csv="/tmp/${oid}_tmpl_thick${thick_str}_init_$$.csv"
  make_init_csv "$oid" "$thick_str" "$fitted_csv" "$init_csv"
  # Fall back to template grid if init failed
  [ -f "$init_csv" ] || init_csv="$fitted_csv"

  {
    echo ">>> Starting TEMPLATE-MCMC PRODUCTION for $oid thick=$thick at $(date)"
    echo "PRODUCTION MODE: nsteps=$NSTEPS, burnin=$BURNIN"
    echo "workdir=$workdir"
    echo "cachedir=$cachedir"
    echo "sed_pkl=$sed_pkl"
    echo "grid-csv=$grid_csv"
    echo "fitted_grid_csv=$init_csv  (bb-initialized)" 
    echo "shell_thickness=$thick"
    echo "emulator_path=$emulator_path"
    echo "sample_log10a=False (analytic)"
    echo "use_weights=True"
    echo "err_floor_frac=0.1"
    echo

    stdbuf -oL -eL python -m lltypeiip.inference.run_sed_mcmc "$oid" \
      --sed-pkl "$sed_pkl" \
      --template-path "$TEMPLATE_PATH" \
      --template-tag "$TEMPLATE_TAG" \
      --fitted-grid-csv "$init_csv" \
      --grid-csv "$grid_csv" \
      --tstar-dummy 6000 \
      --shell-thickness "$thick" \
      --mode data \
      --nsteps "$NSTEPS" \
      --burnin "$BURNIN" \
      --nwalkers 64 \
      --ncores "$ncores" \
      --mp fork \
      --progress-every 100000 \
      --workdir "$workdir" \
      --cache-dir "$cachedir" \
      --cache-ndigits 4 \
      --cache-max 10000 \
      --evaluator-mode emulator \
      --emulator-path "$emulator_path" \
      --err-floor-frac 0.1 \
      --sample-log10a \
      --seed "$SEED"

    local ec=$?

    # Clean up temp init CSV
    rm -f "$init_csv"

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