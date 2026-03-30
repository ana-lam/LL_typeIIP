#!/usr/bin/env python3
"""
Collect log10_a estimates across all model combinations (blackbody/template, thick=2.0/5.0) 
for each oid and compute systematic uncertainty.
   
    python -m lltypeiip.inference.collect_scale_systematics
    python -m lltypeiip.inference.collect_scale_systematics --out-dir mcmc_results/summaries --template-seed 303
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from lltypeiip.config import PROJECT_ROOT

# defaults
DEFAULT_FITTED_GRID_DIR = PROJECT_ROOT / "fitted_grids"
DEFAULT_MCMC_SUMMARY_DIR = PROJECT_ROOT / "mcmc_results/summaries"
DEFAULT_OUT_DIR = PROJECT_ROOT / "scale_results"

def load_mcmc_summary(mcmc_summary_dir, mode, seed=None):
    d = Path(mcmc_summary_dir)
    if seed is not None:
        path = d / f"mcmc_summary_{mode}_seed{seed}.csv"
    else:
        path = d / f"mcmc_summary_{mode}.csv"
    
    if not path.exists():
        for candidate in sorted(d.glob(f"mcmc_summary*{mode}*.csv")):
            path = candidate
            break
    
    if not path.exists():
        print(f"  [warn] No MCMC summary found for mode={mode} (tried {path})")
        return None
    
    print(f"  Loading MCMC summary: {path}")
    return pd.read_csv(path)

def load_fitted_grid_best(fitted_grid_dir, mode, thickness):
    thick_str = str(thickness).replace(".", "_")
    path = (Path(fitted_grid_dir) / mode / f"thick_{thick_str}"
            / f"all_objects_{mode}_thick{thick_str}_fitted.csv")

    if not path.exists():
        print(f"  [warn] Fitted grid CSV not found: {path}")
        return None

    print(f"  Loading fitted grid: {path}")
    df = pd.read_csv(path)

    # keep only best chi2_red row per object
    df = df.sort_values("chi2_red")
    df = df.groupby("oid", as_index=False).first()
    df["log10_a"] = np.log10(df["scale"])
    df["log10_tau_fit"] = np.log10(df["tau"])
    return df[["oid", "log10_a", "log10_tau_fit", "tdust", "chi2_red"]].copy()

def main():

    parser = argparse.ArgumentParser(description="Collect log10_a estimates across all model combinations and compute systematics")
    
    parser.add_argument("--fitted-grid-dir", type=str, default=DEFAULT_FITTED_GRID_DIR,
                        help="Base directory where fitted grid CSVs are stored (organized by mode/thickness).")
    parser.add_argument("--mcmc-summary-dir", type=str, default=DEFAULT_MCMC_SUMMARY_DIR,
                        help="Directory where MCMC summary CSVs are stored.")
    parser.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR,
                        help="Directory to save the combined results with systematics.")
    parser.add_argument("--template-seed", type=int, default=None)
    parser.add_argument("--bb-seed", type=int, default=None)

    args = parser.parse_args()

    # load MCMC summaries
    print("Loading MCMC summaries...")
    df_tmpl_mcmc = load_mcmc_summary(args.mcmc_summary_dir, "template", seed=args.template_seed)
    df_bb_mcmc = load_mcmc_summary(args.mcmc_summary_dir, "blackbody", seed=args.bb_seed)

    # load fitted grid best fits
    print("\nLoading fitted grid bests...")
    grid_data = {}
    for mode in ["blackbody", "template"]:
        for thick in [2.0, 5.0]:
            key = f"{mode}_thick{thick}"
            grid_data[key] = load_fitted_grid_best(args.fitted_grid_dir, mode, thick)
    
    sed_sample_path = PROJECT_ROOT / "sed_sample.txt"
    with open(sed_sample_path) as f:
        all_oids = sorted(line.rstrip() for line in f if line.strip())
    print(f"Loaded {len(all_oids)} objects from sed_sample.txt")

    rows = []
    for oid in all_oids:
        row = {"oid": oid}
        
        # grid log10_a values
        for mode in ["blackbody", "template"]:
            for thick in [2.0, 5.0]:
                key = f"{mode}_thick{thick}"
                col = f"grid_{mode}_t{int(thick)}_log10_a"
                df_g = grid_data[key]
                if df_g is not None:
                    match = df_g[df_g["oid"] == oid]
                    row[col] = float(match["log10_a"].iloc[0]) if not match.empty else np.nan
                else:
                    row[col] = np.nan
        
        # MCMC log10_a values (scale_analytic_map at MAP T_dust, tau)
        for mode, df_mcmc in [("template", df_tmpl_mcmc), ("blackbody", df_bb_mcmc)]:
            if df_mcmc is None:
                for thick in [2.0, 5.0]:
                    row[f"mcmc_{mode}_t{int(thick)}_log10_a"] = np.nan
                    row[f"mcmc_{mode}_t{int(thick)}_log10_a_lo"] = np.nan
                    row[f"mcmc_{mode}_t{int(thick)}_log10_a_hi"] = np.nan
                continue
            for thick in [2.0, 5.0]:
                col_base = f"mcmc_{mode}_t{int(thick)}"
                match = df_mcmc[
                    (df_mcmc["oid"] == oid) &
                    (df_mcmc["shell_thickness"] == thick)
                ]
                if not match.empty:
                    r = match.iloc[0]
                    # use analytic scale at MAP — most reliable point estimate
                    row[f"{col_base}_log10_a"] = float(np.log10(r["scale_analytic_map"])) \
                                                    if pd.notna(r.get("scale_analytic_map")) \
                                                    else float(r["log10_a_map"])
                    # MCMC statistical uncertainty (16-84th percentile)
                    row[f"{col_base}_log10_a_lo"] = float(r["log10_a_lo"])
                    row[f"{col_base}_log10_a_hi"] = float(r["log10_a_hi"])
                else:
                    row[f"{col_base}_log10_a"] = np.nan
                    row[f"{col_base}_log10_a_lo"] = np.nan
                    row[f"{col_base}_log10_a_hi"] = np.nan
    
        point_cols = [c for c in row if c.endswith("_log10_a") and not c.endswith(("_lo", "_hi"))]
        vals = np.array([row[c] for c in point_cols if np.isfinite(row.get(c, np.nan))])

        if len(vals) >= 2:
            row["log10_a_sys_std"] = float(np.std(vals))
            row["log10_a_sys_halfrange"] = float((np.max(vals) - np.min(vals)) / 2)
            row["log10_a_mean"] = float(np.mean(vals))
            row["n_estimates"] = len(vals)
        else:
            row["log10_a_sys_std"] = np.nan
            row["log10_a_sys_halfrange"] = np.nan
            row["log10_a_mean"] = float(vals[0]) if len(vals) == 1 else np.nan
            row["n_estimates"] = len(vals)

        rows.append(row)
    
    df_out = pd.DataFrame(rows)

    id_cols = ["oid", "n_estimates", "log10_a_mean", "log10_a_sys_std", "log10_a_sys_halfrange"]
    grid_cols = sorted([c for c in df_out.columns if c.startswith("grid_")])
    mcmc_cols = sorted([c for c in df_out.columns if c.startswith("mcmc_")])
    df_out = df_out[id_cols + grid_cols + mcmc_cols]

    out_path = Path(args.out_dir) / "log10_a_systematics.csv"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"\nSaved -> {out_path}")
    print(f"Rows: {len(df_out)}  columns: {list(df_out.columns)}")
    print("\nSample (first 5 objects):")
    print(df_out[["oid", "log10_a_mean", "log10_a_sys_std", "log10_a_sys_halfrange", "n_estimates"]]
          .head().to_string(index=False))

if __name__ == "__main__":
    main()
    