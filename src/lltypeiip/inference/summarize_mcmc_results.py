#!/usr/bin/env python3
"""
Summarize MCMC posterior results into a CSV with median & MAP + chi2 from analytic & MCMC sampled 
log10a.
   
    # Single object, template mode, both thicknesses
    python -m lltypeiip.inference.summarize_mcmc_results ZTF22abtspsw --mode template --thickness both

    # All objects, both modes, both thicknesses
    python -m lltypeiip.inference.summarize_mcmc_results --all --mode both --thickness both --seed 303
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from lltypeiip.config import config, PROJECT_ROOT
from lltypeiip.dusty.runner import DustyRunner
from lltypeiip.dusty.scaling import compute_chi2
from lltypeiip.dusty.custom_spectrum import NugentIIPSeries
from lltypeiip.sed.build import _prepare_sed_xy
from lltypeiip.inference.mcmc import _ls_scale_and_chi2


# --- defaults ---
DEFAULT_MCMC_DIR = PROJECT_ROOT / "mcmc_results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "mcmc_results/summaries"
DEFAULT_SED_DIR = PROJECT_ROOT / "data/tail_seds"
DEFAULT_WORKDIR = Path("/tmp/lltypeiip_mcmc_summary")
DEFAULT_DUSTY_CACHE_DIR = PROJECT_ROOT / config.dusty.npz_cache_dir

TEMPLATE_PATH = PROJECT_ROOT /  "data/typeiip_spectral_templates/sn2p_flux.v1.2.dat"
TEMPLATE_TAG = "nugent_iip"
TSTAR_DUMMY = 6000.0

def _mcmc_tag(mode):
    return "template" if mode == "template" else "bb"

def load_sed(oid, sed_dir=DEFAULT_SED_DIR):
    sed_path = Path(sed_dir) / f"{oid}_tail_sed.pkl"
    if not sed_path.exists():
        raise FileNotFoundError(f"SED not found: {sed_path}")
    with open(sed_path, "rb") as f:
        sed_data = pickle.load(f)
    
    if isinstance(sed_data, dict) and "sed" in sed_data:
        sed = sed_data["sed"]
        if "phase_days" not in sed and "phase" in sed_data:
            sed["phase_days"] = sed_data["phase"]
    else:
        sed = sed_data

    if "oid" not in sed:
        sed["oid"] = oid
    
    return sed

def _compute_chi2_analytic(lam_um, lamFlam, sed, y_mode="Flam"):
    """Compute chi2 for analytic best-fit model. For comparing to grid fits chi2."""
    scale, chi2 = _ls_scale_and_chi2(lam_um, lamFlam, sed, a=None,
                                      y_mode=y_mode, use_weights=True)
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)

    mask = (x_sed >= lam_um.min()) & (x_sed <= lam_um.max())
    ndim = 2  # tdust, log10_tau only — scale is analytic, not a free param
    dof  = max(int(np.sum(mask)) - ndim, 1)
    chi2_red = chi2 / dof
    return float(scale), float(chi2), int(dof), float(chi2_red)

def _compute_chi2_mcmc(lam_um, lamFlam, sed, log10_a, y_mode="Flam"):
    """Chi2 using the MCMC-sampled log10_a value."""
    a = 10.0 ** log10_a
    scale, chi2 = _ls_scale_and_chi2(lam_um, lamFlam, sed, a=a,
                                      y_mode=y_mode, use_weights=True)
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)
    mask = (x_sed >= lam_um.min()) & (x_sed <= lam_um.max())
    ndim = 3
    dof  = max(int(np.sum(mask)) - ndim, 1)
    chi2_red = chi2 / dof
    return float(a), float(chi2), int(dof), float(chi2_red)

def summarize_mcmc_results(oid, mode, thickness, mcmc_dir=DEFAULT_MCMC_DIR, 
                          out_dir=DEFAULT_OUTPUT_DIR, sed_dir=DEFAULT_SED_DIR, 
                          workdir=DEFAULT_WORKDIR, cache_dir=DEFAULT_DUSTY_CACHE_DIR,
                          mcmc_mode="mixture", seed=None, y_mode="Flam"):
    """
    Summarize MCMC results for a single object, mode, and thickness. 
    Returns list of dicts for median row, MAP row.
    """

    thick_str = str(thickness).replace('.', '_')
    tag = _mcmc_tag(mode)

    if seed is None:
        mcmc_path = Path(mcmc_dir) / oid / f"mcmc_{oid}_{tag}_thick{thick_str}_{mcmc_mode}.npz"
    else:
        mcmc_path = Path(mcmc_dir) / oid / f"mcmc_{oid}_{tag}_thick{thick_str}_{mcmc_mode}_seed{seed}.npz"

    try:
        sed = load_sed(oid, sed_dir=sed_dir)
    except FileNotFoundError as e:
        print(f"Error loading SED for {oid}: {e}")
        return []

    d = np.load(mcmc_path, allow_pickle=True)
    samples = d["samples"]
    log_prob = d["log_prob"]
    grid_best = d["grid_best"].item()

    if tag == "template":
        tdust_samples = samples[:, 0]
        log10_tau_samples = samples[:, 1]
        log10_a_samples = samples[:, 2]
        tstar_samples = np.full(len(samples), TSTAR_DUMMY)
    else:
        tstar_samples = samples[:, 0]
        tdust_samples = samples[:, 1]
        log10_tau_samples = samples[:, 2]
        log10_a_samples = samples[:, 3]
    
    def _stats(vals):
        med = np.median(vals)
        lo = np.percentile(vals, 16)
        hi = np.percentile(vals, 84) - med
        return med, lo, hi
    
    tdust_med, tdust_lo, tdust_hi = _stats(tdust_samples)
    log10_tau_med, log10_tau_lo, log10_tau_hi = _stats(log10_tau_samples)
    log10_a_med, log10_a_lo, log10_a_hi = _stats(log10_a_samples)

    if tag != "template":
        tstar_med, tstar_lo, tstar_hi = _stats(tstar_samples)

    # MAP sample (highest log probability)
    map_idx = np.argmax(log_prob)
    tdust_map = tdust_samples[map_idx]
    log10_tau_map = log10_tau_samples[map_idx]
    log10_a_map = log10_a_samples[map_idx]
    tstar_map = tstar_samples[map_idx]

    # build DUSTY runner
    runner = DustyRunner(
        dusty_file_dir=config.dusty.dusty_file_dir,
        base_workdir=workdir,
        cache_dir=cache_dir,
        shell_thickness=thickness,
        cache_max=2000,
        use_tmp=True,
        run_tag=f"summary_{oid}_{tag}_thick{thick_str}",
    )

    template = None
    phase_days = None
    if mode == "template":
        template = NugentIIPSeries(str(TEMPLATE_PATH.resolve()))
        phase_days = float(sed.get("phase_days", sed.get("phase", 0)))

    # compute both chi2
    def _evaluate(tdust, log10_tau, log10_a, tstar):
        tau = 10.0 ** log10_tau
        lam_um, lamFlam, r1 = runner.evaluate_model(
            tstar=tstar,
            tdust=tdust,
            tau=tau,
            shell_thickness=thickness,
            template=template,
            phase_days=phase_days,
            template_tag=TEMPLATE_TAG,
        )
        if lam_um is None:
            return None
        
        scale_analytic, chi2_analytic, dof_analytic, chi2_red_analytic = _compute_chi2_analytic(
            lam_um, lamFlam, sed, y_mode=y_mode
        )
        _, chi2_mcmc, _, chi2_red_mcmc = _compute_chi2_mcmc(
            lam_um, lamFlam, sed, log10_a, y_mode=y_mode
        )

        return dict(
            lam_um=lam_um,
            lamFlam=lamFlam,
            scale_analytic=scale_analytic,
            log10_a_mcmc = log10_a,
            chi2_analytic=chi2_analytic,
            chi2_red_analytic=chi2_red_analytic,
            chi2_mcmc=chi2_mcmc,
            chi2_red_mcmc=chi2_red_mcmc,
            dof = dof_analytic,
            tau = tau,
            r1 = r1
        )

    # ---- evaluate at median ----
    med_res = _evaluate(tdust_med, log10_tau_med, log10_a_med,
                        TSTAR_DUMMY if tag == "template" else tstar_med)
    # ---- evaluate at MAP ----
    map_res = _evaluate(tdust_map, log10_tau_map, log10_a_map,
                        TSTAR_DUMMY if tag == "template" else tstar_map)

    if med_res is not None:
        print(f"    median: Tdust={tdust_med:.0f}K  log10τ={log10_tau_med:.2f}  "
              f"χ²_red(ana)={med_res['chi2_red_analytic']:.2f}  "
              f"χ²_red(mcmc)={med_res['chi2_red_mcmc']:.2f}")
    else:
        print(f"    median: DUSTY failed")

    if map_res is not None:
        print(f"    MAP:    Tdust={tdust_map:.0f}K  log10τ={log10_tau_map:.2f}  "
              f"χ²_red(ana)={map_res['chi2_red_analytic']:.2f}  "
              f"χ²_red(mcmc)={map_res['chi2_red_mcmc']:.2f}")
    else:
        print(f"    MAP: DUSTY failed")

    # ---- build single row ----
    row = dict(
        oid = oid,
        mode = mode,
        shell_thickness = thickness,
        phase_days = sed.get("phase_days", None),
        mjd = sed.get("mjd", None),
        mcmc_mode = mcmc_mode,
        seed = seed,
        n_samples = len(samples),
        # posterior parameter summaries
        tdust_med = tdust_med,
        tdust_lo = tdust_lo,
        tdust_hi = tdust_hi,
        log10_tau_med = log10_tau_med,
        log10_tau_lo = log10_tau_lo,
        log10_tau_hi = log10_tau_hi,
        log10_a_med = log10_a_med,
        log10_a_lo = log10_a_lo,
        log10_a_hi = log10_a_hi,
        tau_med = 10.0 ** log10_tau_med,
        # MAP parameter point
        tdust_map = tdust_map,
        log10_tau_map = log10_tau_map,
        log10_a_map = log10_a_map,
        tau_map = 10.0 ** log10_tau_map,
        # chi2 at median
        scale_analytic_med = med_res["scale_analytic"] if med_res else None,
        chi2_analytic_med = med_res["chi2_analytic"] if med_res else None,
        chi2_red_analytic_med = med_res["chi2_red_analytic"] if med_res else None,
        chi2_mcmc_med = med_res["chi2_mcmc"] if med_res else None,
        chi2_red_mcmc_med  = med_res["chi2_red_mcmc"] if med_res else None,
        dof = med_res["dof"] if med_res else None,
        # chi2 at MAP
        scale_analytic_map = map_res["scale_analytic"] if map_res else None,
        chi2_analytic_map = map_res["chi2_analytic"] if map_res else None,
        chi2_red_analytic_map = map_res["chi2_red_analytic"] if map_res else None,
        chi2_mcmc_map = map_res["chi2_mcmc"] if map_res else None,
        chi2_red_mcmc_map = map_res["chi2_red_mcmc"] if map_res else None,
        # grid best for reference
        grid_tdust = grid_best.get("tdust"),
        grid_log10_tau = grid_best.get("log10_tau"),
        grid_log10_a = grid_best.get("log10_a"),
    )

    if tag != "template":
        row.update(dict(
            tstar_med = tstar_med,
            tstar_lo = tstar_lo,
            tstar_hi = tstar_hi,
            tstar_map = tstar_map,
        ))
    else:
        row["tstar_dummy"] = TSTAR_DUMMY

    return row

def main():
    parser = argparse.ArgumentParser(
        description="Summarize MCMC results into CSV."
    )

    parser.add_argument("oid", nargs="?", help="Object ID (e.g. ZTF22abtspsw)")
    parser.add_argument("--all", action="store_true",
                        help="Process all objects with MCMC results")
    parser.add_argument("--mode", choices=["blackbody", "template", "both"],
                        default="both")
    parser.add_argument("--thickness",
                        help="Shell thickness: 2.0, 5.0, or 'both'", default="both")
    parser.add_argument("--seed", type=int,  default=None)
    parser.add_argument("--mcmc-mode", default="mixture")
    parser.add_argument("--mcmc-dir", default=str(DEFAULT_MCMC_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sed-dir", default=str(DEFAULT_SED_DIR))
    parser.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_DUSTY_CACHE_DIR))

    args = parser.parse_args()

    mcmc_dir = Path(args.mcmc_dir)
    out_dir = Path(args.out_dir)
    sed_dir = Path(args.sed_dir)
    workdir = Path(args.workdir)
    cache_dir = Path(args.cache_dir)

    thicknesses = [2.0, 5.0] if args.thickness == "both" else [float(args.thickness)]
    modes = ["blackbody", "template"] if args.mode == "both" else [args.mode]

    if args.all:
        sed_sample_path = PROJECT_ROOT / "sed_sample.txt"
        with open(sed_sample_path, 'r') as file:
            oids = [line.rstrip() for line in file]
        print(f"Found {len(oids)} objects in sample list")
    elif args.oid:
        oids = [args.oid]
    else:
        parser.print_help()
        return

    all_rows = []

    for oid in oids:
        print(f"\n{'='*60}")
        print(f"OID: {oid}")
        for mode in modes:
            for thickness in thicknesses:
                print(f"  {mode} thick={thickness}")
                row = summarize_mcmc_results(
                    oid=oid, mode=mode, thickness=thickness,
                    mcmc_dir=mcmc_dir, out_dir=out_dir,
                    sed_dir=sed_dir, workdir=workdir,
                    cache_dir=cache_dir,
                    mcmc_mode=args.mcmc_mode,
                    seed=args.seed,
                )
                if row is not None:
                    all_rows.append(row)

    if not all_rows:
        print("No results to save.")
        return

    df = pd.DataFrame(all_rows)

    # ---- column ordering ----
    first_cols = ["oid", "mode", "shell_thickness", "phase_days", "mjd",
                   "mcmc_mode", "seed", "n_samples"]
    med_param = ["tdust_med", "tdust_lo", "tdust_hi",
                   "log10_tau_med", "log10_tau_lo", "log10_tau_hi",
                   "log10_a_med", "log10_a_lo", "log10_a_hi",
                   "tau_med", "tstar_med", "tstar_lo", "tstar_hi", "tstar_dummy"]
    map_param = ["tdust_map", "log10_tau_map", "log10_a_map", "tau_map", "tstar_map"]
    med_fit = ["scale_analytic_med", "chi2_analytic_med", "chi2_red_analytic_med",
                   "chi2_mcmc_med", "chi2_red_mcmc_med", "dof"]
    map_fit = ["scale_analytic_map", "chi2_analytic_map", "chi2_red_analytic_map",
                   "chi2_mcmc_map", "chi2_red_mcmc_map"]
    grid_cols = ["grid_tdust", "grid_log10_tau", "grid_log10_a"]

    ordered = []
    for c in first_cols + med_param + map_param + med_fit + map_fit + grid_cols:
        if c in df.columns:
            ordered.append(c)
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    # --- save ---
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "both":
        if args.seed is not None:
            out_path = out_dir / f"mcmc_summary_seed{args.seed}.csv"
        else:
            out_path = out_dir / "mcmc_summary.csv"
    else:
        if args.seed is not None:
            out_path = out_dir / f"mcmc_summary_{mode}_seed{args.seed}.csv"
        else:
            out_path = out_dir / f"mcmc_summary_{mode}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Saved summary -> {out_path}")
    print(f"Rows: {len(df)}  ({df['oid'].nunique()} objects)")
    print(f"\nchi2_red_analytic (median) per object:")
    for _, row in df.iterrows():
        print(f"  {row['oid']} {row['mode']} thick={row['shell_thickness']}: "
              f"χ²_red(med)={row['chi2_red_analytic_med']:.2f}  "
              f"χ²_red(MAP)={row['chi2_red_analytic_map']:.2f}")

if __name__ == "__main__":
    main()
    
