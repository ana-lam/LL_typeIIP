#!/usr/bin/env python3
"""
Create fitted grid summary CSVs with chi-squared values for each SED.

    # Fit blackbody grid to one SED
    python create_fitted_grid_summaries.py ZTF22abtspsw --mode blackbody --thickness 2.0
    
    # Fit template grid to one SED
    python create_fitted_grid_summaries.py ZTF22abtspsw --mode template --thickness 2.0
    
    # Fit grids to all SEDs
    python create_fitted_grid_summaries.py --all --mode blackbody --thickness 2.0
    python create_fitted_grid_summaries.py --all --mode template --thickness 2.0
    
    # Both modes, both thicknesses, all SEDs
    python create_fitted_grid_summaries.py --all --mode both --thickness both
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from lltypeiip.dusty import fit_grid_to_sed

def load_sed(oid, sed_dir="data/tail_seds"):
    """Load SED for given OID."""
    sed_path = Path(sed_dir) / f"{oid}_tail_sed.pkl"

    if not sed_path.exists():
        raise FileNotFoundError(f"SED not found: {sed_path}")
    
    with open(sed_path, "rb") as f:
        sed_data = pickle.load(f)
    
    # unwrap if needed
    if isinstance(sed_data, dict) and "sed" in sed_data:
        sed = sed_data["sed"]
        if "phase_days" not in sed and "phase" in sed_data:
            sed["phase_days"] = sed_data["phase"]
    else:
        sed = sed_data

    if "oid" not in sed:
        sed["oid"] = oid
    
    return sed

def create_fitted_grid_summary(oid, mode, thickness, sed_dir="data/tail_seds",
                               output_dir="fitted_grids", max_tstar=6000.0):
    """
    Fit grid to SED and save summary CSV.

    Parameters
    ----------
    oid : str
        Object ID
    mode : str
        'blackbody' or 'template'
    thickness : float
        Shell thickness (2.0 or 5.0)
    sed_dir : str
        Directory containing SED pickles
    output_dir : str
        Directory to save fitted grid summaries
    
    Returns
    -------
    output_path : Path
        Path to saved fitted grid CSV
    """

    # load SED
    print(f"Loading SED for {oid}...")
    sed = load_sed(oid, sed_dir=sed_dir)

    # construct grid CSV path
    thick_str = str(thickness).replace('.', '_')

    if mode == 'blackbody':
        grid_csv = f"dusty_runs/blackbody_grids/silicate_tau_0.55um_thick_{thick_str}/grid_summary_blackbody_thick_{thick_str}.csv"
    else:  # template
        grid_csv = f"dusty_runs/template_grids/silicate_tau_0.55um_thick_{thick_str}/grid_summary_nugent_iip_thick_{thick_str}.csv"

    grid_csv = Path(grid_csv)
    if not grid_csv.exists():
        raise FileNotFoundError(f"Grid CSV not found: {grid_csv}")
    
    print(f"Fitting {mode} grid (thickness={thickness}) to {oid}...")

    # Fit grid to SED
    df_fitted = fit_grid_to_sed(
        grid_csv=str(grid_csv),
        sed=sed,
        shell_thickness=thickness,
        y_mode="Flam",
        use_weights=True,
        max_tstar=max_tstar if mode == 'blackbody' else None
    )

    df_fitted['oid'] = oid
    df_fitted['mode'] = mode

    if 'phase_days' in sed:
        df_fitted['phase_days'] = sed['phase_days']
    
    if 'mjd' in sed:
        df_fitted['mjd'] = sed['mjd']

    first_cols = ['oid', 'mode', 'shell_thickness']
    if 'phase_days' in df_fitted.columns:
        first_cols.append('phase_days')
    if 'mjd' in df_fitted.columns:
        first_cols.append('mjd')

    if mode == 'blackbody':
        param_cols = ['tstar', 'tdust', 'tau']
    else:
        param_cols = ['tdust', 'tau']
        if 'template_tag' in df_fitted.columns:
            first_cols.append('template_tag')

    fit_cols = ['scale', 'chi2', 'dof', 'chi2_red']

    path_cols = []
    if 'npz_path' in df_fitted.columns:
        path_cols.append('npz_path')
    if 'outpath' in df_fitted.columns:
        path_cols.append('outpath')

    other_cols = [c for c in df_fitted.columns 
                  if c not in first_cols + param_cols + fit_cols + path_cols]
    
    df_fitted = df_fitted[first_cols + param_cols + fit_cols + path_cols + other_cols]

    # create output directory
    output_dir = Path(output_dir) / mode / f"thick_{thick_str}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save fitted grid
    output_filename = f"{oid}_{mode}_thick{thick_str}_fitted.csv"
    output_path = output_dir / output_filename
    
    df_fitted.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    
    best = df_fitted.iloc[0]
    if mode == 'blackbody':
        print(f"χ²_red={best['chi2_red']:.2f} | Tstar={best['tstar']:.0f}K, Tdust={best['tdust']:.0f}K, τ={best['tau']:.3f}")
    else:  # template
        print(f"χ²_red={best['chi2_red']:.2f} | Tstar={best['tstar_dummy']:.0f}K, Tdust={best['tdust']:.0f}K, τ={best['tau']:.3f}")
    
    return output_path

def create_combined_summary(oids, mode, thickness, sed_dir="data/tail_seds", output_dir="fitted_grids"):
    """
    Create combined summary CSV for multiple OIDs.
    """

    all_results = []

    for oid in tqdm(oids, desc=f"Fitting {mode} grids"):
        try:
            # create individual fitted grid
            output_path = create_fitted_grid_summary(
                oid, mode, thickness, sed_dir, output_dir
            )
            
            # load the fitted results
            df = pd.read_csv(output_path)
            all_results.append(df)
            
        except Exception as e:
            print(f"Error fitting {oid}: {e}")
            continue
    
    if not all_results:
        print("No successful fits!")
        return None
    
    # combine all results
    df_combined = pd.concat(all_results, ignore_index=True)
    
    # save combined CSV
    thick_str = str(thickness).replace('.', '_')
    output_dir = Path(output_dir) / mode / f"thick_{thick_str}"
    combined_path = output_dir / f"all_objects_{mode}_thick{thick_str}_fitted.csv"
    
    df_combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined summary: {combined_path}")
    print(f"Total objects: {len(oids)}")
    print(f"Successful fits: {df_combined['oid'].nunique()}")
    
    # print summary statistics
    print("\nSummary Statistics:")
    print(f"  Mean χ²_red: {df_combined.groupby('oid')['chi2_red'].min().mean():.2f}")
    print(f"  Median χ²_red: {df_combined.groupby('oid')['chi2_red'].min().median():.2f}")
    
    # print best fit for each object
    print("\nBest fits per object:")
    best_per_oid = df_combined.loc[df_combined.groupby('oid')['chi2_red'].idxmin()]
    for _, row in best_per_oid.iterrows():
        if mode == 'blackbody':
            print(f"  {row['oid']}: Tstar={row['tstar']:.0f}K, Tdust={row['tdust']:.0f}K, "
                  f"τ={row['tau']:.3f}, χ²_red={row['chi2_red']:.2f}")
        else:
            print(f"  {row['oid']}: Tstar={row['tstar_dummy']:.0f}K, Tdust={row['tdust']:.0f}K, τ={row['tau']:.3f}, "
                  f"χ²_red={row['chi2_red']:.2f}")
    
    return combined_path

def main():
    parser = argparse.ArgumentParser(
        description="Create fitted grid summary CSVs with chi-squared values"
    )

    parser.add_argument("oid", nargs='?', help="Object ID (e.g., ZTF22abtspsw)")
    parser.add_argument("--all", action='store_true', 
                       help="Process all objects in sed_dir")
    parser.add_argument("--mode", choices=['blackbody', 'template', 'both'],
                       default='both', help="Which model(s) to fit")
    parser.add_argument("--thickness", 
                       help="Shell thickness: 2.0, 5.0, or 'both'")
    parser.add_argument("--sed-dir", default="data/tail_seds",
                       help="Directory containing SED pickles")
    parser.add_argument("--output-dir", default="fitted_grids",
                       help="Directory to save fitted grid summaries")
    
    args = parser.parse_args()

    # thickness values
    if args.thickness == 'both':
        thicknesses = [2.0, 5.0]
    else:
        thicknesses = [float(args.thickness)]

    # model modes
    if args.mode == 'both':
        modes = ['blackbody', 'template']
    else:
        modes = [args.mode]

    # get list of OIDs
    if args.all:
        sed_dir = Path(args.sed_dir)
        sed_files = sorted(sed_dir.glob("*_tail_sed.pkl"))
        oids = [f.stem.replace("_tail_sed", "") for f in sed_files]
        print(f"Found {len(oids)} objects with SEDs")
    elif args.oid:
        oids = [args.oid]
    else:
        parser.print_help()
        return
    
    # process each mode and thickness
    for mode in modes:
        for thickness in thicknesses:
            print(f"\n{'='*70}")
            print(f"Processing {mode.upper()} mode with thickness={thickness}")
            print(f"{'='*70}\n")
            
            if len(oids) == 1:
                # Single object
                create_fitted_grid_summary(
                    oids[0], mode, thickness, 
                    args.sed_dir, args.output_dir
                )
            else:
                # create combined summary
                create_combined_summary(
                    oids, mode, thickness,
                    args.sed_dir, args.output_dir
                )
    
    print("\n" + "="*70)
    print("Done! Fitted grid summaries saved to:", args.output_dir)
    print("="*70)

if __name__ == "__main__":
    main()