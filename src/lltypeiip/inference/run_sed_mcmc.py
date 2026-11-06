# Run with:
# python -m lltypeiip.inference.run_sed_mcmc <OID>

import argparse
import os
import numpy as np
from pathlib import Path
import pickle
import pandas as pd

from ..sed import build_multi_epoch_seds_from_tail
from ..sed.build import _unwrap_sed
from .mcmc import run_mcmc_for_sed
from ..config import config, PROJECT_ROOT
from ..photometry import get_wise_lc_data, get_ztf_lc_data, convert_ZTF_mag_mJy
from ..dusty import fit_grid_to_sed, NugentIIPSeries, DustyRunner

from alerce.core import Alerce

default_workdir = config.dusty.workdir_mcmc_dir
default_outdir  = getattr(config.paths, "mcmc_results_dir", str(PROJECT_ROOT / "mcmc_results"))

def _load_spectrum(spectrum_path, z=0.0):
    """Load a .flm spectrum, de-redshift, and package for MCMC fitting."""
    data = np.loadtxt(spectrum_path, comments="#")
    wl_obs = data[:, 0]
    flux = data[:, 1] * 1e-15  # -> erg/s/cm^2/A (de Jaeger et al. 2019 convention)

    wl_rest_A = wl_obs / (1 + z)
    wl_rest_um = wl_rest_A * 1e-4
    lamFlam = wl_rest_A * flux  # lambda * F_lambda, matches model.lamFlam units

    order = np.argsort(wl_rest_um)
    return {"lam_um": wl_rest_um[order], "flux": lamFlam[order]}

def parse_args():

    p = argparse.ArgumentParser(description="Run DUSTY MCMC for SN SED")

    # required
    p.add_argument("oid", type=str, help="ZTF object ID (e.g. ZTF22abtspsw)")

    p.add_argument("--run-id", type=str, default="",
               help="Optional suffix to distinguish runs (e.g. 'no_i_band'). "
                    "Appended to output filenames and run tags.")

    # template mode
    p.add_argument("--template-path", type=str, default=None,
                   help="If set: use Spectrum=5 template mode.")
    p.add_argument("--template-tag", type=str, default="nugent_iip")
    p.add_argument("--grid-csv", type=str, default=None,
                   help="CSV from running grids, used to pick best row for initialization.")
    p.add_argument("--fitted-grid-csv", type=str, default=None,
               help="Pre-fitted grid CSV (from fit_grids.py). "
                    "If provided, skips re-fitting and uses this directly.")
    p.add_argument("--tstar-dummy", type=float, default=6000.0)

    p.add_argument("--template-kind", type=str, default="auto",
               choices=["auto", "series", "single_spectrum"],
               help="'series' for Nugent-style multi-phase grid file, "
                    "'single_spectrum' for a single .flm spectrum, "
                    "'auto' infers from file extension.")
    p.add_argument("--spectrum-z", type=float, default=0.0,
                help="Redshift for de-redshifting a single-spectrum template "
                        "(only used when template-kind=single_spectrum).")

    # optionally load one saved sed pickle instead of rebuilding
    p.add_argument("--sed-pkl", type=str, default=None,
                   help="Optional: path to a saved tail_sed pickle. Can be plain SED dict or payload with key 'sed'.")
    
    # evaluator mode
    p.add_argument("--evaluator-mode", choices=["emulator", "dusty"], default="dusty",
                   help="Mode for evaluating the model: 'emulator' or 'dusty'.")
    p.add_argument("--emulator-path", type=str, default=None,
                   help="Path to the emulator file (required if using 'emulator' mode).")

    # sampler params
    p.add_argument("--sweep", action="store_true",
               help="Run data, anchored, and mixture back-to-back for comparison")
    p.add_argument("--mode", choices=["data", "anchored", "mixture"],
                   default="data", help="Posterior mode")
    p.add_argument("--nwalkers", type=int, default=32)
    p.add_argument("--nsteps", type=int, default=4000)
    p.add_argument("--burnin", type=int, default=1000)
    p.add_argument("--ncores", type=int, default=4)
    p.add_argument("--progress-every", type=int, default=100)

    # log10_a sampling mode
    p.add_argument("--sample-log10a", action="store_true", default=False,
                   help="Whether to sample log10_a or compute it analytically.")
    p.add_argument("--log10a-range", type=float, default=3.0)

    # mixture
    p.add_argument("--mix-weight", type=float, default=0.3,
                   help="Gaussian weight for mixture prior")

    p.add_argument("--workdir", type=str, default=default_workdir)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mp", choices=["spawn", "fork"], default="spawn")

    # cache
    p.add_argument("--cache-dir", type=str, default=None,
               help="On-disk cache for DUSTY results (npz). Default: <workdir>/dusty_npz_cache")
    p.add_argument("--cache-ndigits", type=int, default=4,
                help="Rounding digits used for tau/thickness cache keys.")
    p.add_argument("--cache-max", type=int, default=5000,
                help="Max number of models kept in memory cache per process.")
    p.add_argument("--no-tmp", action="store_true", help="Disable temp dirs for DUSTY runs")

    # shell thickness
    p.add_argument("--shell-thickness", type=float, default=2.0,
                   help="Shell thickness (Y_out/Y_in). Default: 2.0")
    
    # no weights for chi2 fitting
    p.add_argument("--no-weights", action="store_true",
                   help="Disable weights for chi2 fitting (i.e. use unweighted residuals)")
    p.add_argument("--err-floor-frac", type=float, default=0.0,
                   help="Fractional error floor to add in quadrature to observed flux uncertainties. Default: 0.0 (no error floor)")
    p.add_argument("--equal-weights", action="store_true", default=False,
                   help="Use equal weights for all data points in chi2")
    p.add_argument("--quadratic-weights", action="store_true", default=False,
                   help="Use quadratic weights for chi2 (adds error floor in quadrature)")

    # spectrum fitting
    p.add_argument("--spectrum-path", type=str, default=None,
               help="Optional path to a .flm spectrum to include in the MCMC "
                    "likelihood alongside photometry (template mode only).")
    p.add_argument("--spectrum-fit-z", type=float, default=0.0,
                help="Redshift for de-redshifting --spectrum-path before "
                        "fitting (distinct from --spectrum-z, which is for "
                        "single_spectrum template mode).")
    p.add_argument("--spectrum-err-floor-frac", type=float, default=0.1,
                help="Fractional error floor per spectral bin.")
    p.add_argument("--spectrum-weight-scale", type=float, default=1.0,
               help="Multiplier on the spectral term's contribution to the "
                    "GRID-SEEDING fit only (fit_grid_to_sed). The actual MCMC "
                    "likelihood automatically rebalances W1/W2/spectrum to "
                    "contribute equally via compute_group_weight_scales() and "
                    "does not use this value.")


    return p.parse_args()

def main():
    
    args = parse_args()

    oid = args.oid

    template_mode = (args.template_path is not None)

    if args.cache_dir is None:
        if template_mode:
            args.cache_dir = str(config.dusty.template_npz_cache_dir)
        else:
            args.cache_dir = str(config.dusty.blackbody_npz_cache_dir)

    if args.no_weights:
        use_weights = False
        print("Running with unweighted chi2 (no weights).")
    else:
        use_weights = True

    print(f"Running MCMC for OID: {oid}")
    print(f"Posterior mode: {args.mode}")
    print(f"Shell thickness: {args.shell_thickness}")

    dusty_file_dir = config.dusty.dusty_file_dir
    workdir = args.workdir

    os.makedirs(workdir, exist_ok=True)

    print(f"DUSTY binary dir: {dusty_file_dir}")
    print(f"Working directory: {workdir}")

    # photometry stuff
    if args.sed_pkl:
        with open(Path(args.sed_pkl).resolve(), "rb") as f:
            obj = pickle.load(f)
        sed = _unwrap_sed(obj)
        if sed.get("oid", oid) != oid:
            print(f"[warn] sed oid={sed.get('oid')} but CLI oid={oid}")
        if "phase_days" not in sed and template_mode:
            raise RuntimeError("Template mode needs sed['phase_days']. Your pickle doesn't include it.")
    else:
        alerce = Alerce()

        wise_resdict = get_wise_lc_data(oid)
        ztf_resdict = get_ztf_lc_data(
            oid, alerce, doLC=False, doStamps=False, add_forced=True
        )
        ztf_resdict = convert_ZTF_mag_mJy(ztf_resdict, forced=True)

        # sed stuff
        seds = build_multi_epoch_seds_from_tail(
            ztf_resdict,
            wise_resdict,
            min_detected_bands=4,
            require_wise_detection=True,
            max_dt_ztf=5.0,
            max_dt_wise=5.0
        )

        if len(seds) == 0:
            raise RuntimeError("No valid SEDs constructed (WISE detection required).")

        if len(seds) > 0:
            print(f"  Found {len(seds)} SED tail epochs with WISE detections for {oid}:")
            print("\n".join(
                [f"  MJD {sed['mjd']:.2f}, bands: {list(sed['bands'])}" for sed in seds]
            ))
            best = max(seds, key=lambda sed: (len(sed["bands"]), sed["mjd"]))
            sed = best
    
    print(f"Using SED epoch: oid={sed.get('oid')} mjd={float(sed['mjd']):.3f} bands={sed.get('bands')}")
    if template_mode:
        print(f"Template mode phase_days={sed.get('phase_days')} template={args.template_tag}")

    spectrum = None
    if args.spectrum_path is not None:
        print(f"Loading spectrum from {args.spectrum_path} (z={args.spectrum_fit_z})...")
        spectrum = _load_spectrum(args.spectrum_path, z=args.spectrum_fit_z)

    # grid fitting
    if args.fitted_grid_csv is not None:
        fitted_csv = Path(args.fitted_grid_csv).resolve()
        if not fitted_csv.exists():
            raise RuntimeError(f"Fitted grid CSV not found: {fitted_csv}")
        print(f"Loading pre-fitted grid from: {fitted_csv}")
        df = pd.read_csv(fitted_csv)
        df_oid = df[df["oid"] == oid] if "oid" in df.columns else df
        if df_oid.empty:
            raise RuntimeError(f"No rows for oid={oid} in {fitted_csv}")
        df = df_oid.sort_values("chi2_red").reset_index(drop=True)
        print(f"Loaded {len(df)} pre-fitted models for {oid}")
    elif args.grid_csv is not None:
        grid_csv = Path(args.grid_csv).resolve()
        if not grid_csv.exists():
            raise RuntimeError(f"Grid CSV not found: {grid_csv}")
        print(f"Fitting grid from: {grid_csv}")
        df = fit_grid_to_sed(
            grid_csv=str(grid_csv),
            sed=sed,
            shell_thickness=args.shell_thickness,
            template_tag=args.template_tag if template_mode else None,
            y_mode="Flam",
            use_weights=True,
            top_k=None,
            spectrum=spectrum,
            spectrum_err_floor_frac=args.spectrum_err_floor_frac,
        )
    else:
        raise RuntimeError("Requires either --grid-csv or --fitted-grid-csv")

    tstar = df.iloc[0]['tstar'] if 'tstar' in df.columns else df.iloc[0]['tstar_dummy'] if 'tstar_dummy' in df.columns else None
    print(f"Loaded {len(df)} grid models")
    print(f"Best fit: tstar={tstar}K, tdust={df.iloc[0]['tdust']:.1f}K, tau={df.iloc[0]['tau']:.4f}, χ²_red={df.iloc[0]['chi2_red']:.2f}")

    shell_thickness = args.shell_thickness
    dust_type = "silicate"

    print("\nStarting MCMC...")
    print(f"mp_prefer={args.mp}  ncores={args.ncores}  nwalkers={args.nwalkers}", flush=True)

    modes = [args.mode] if not args.sweep else ["data", "anchored", "mixture"]
    mode_kwargs = {
        "data": dict(
            posterior_mode="data",
            init_mode="hybrid",
            init_scales={"tstar_frac": 0.1, "tdust_frac": 0.1, "log10_tau": 0.3, "log10_a": 0.3},
        ),
        "anchored": dict(
            posterior_mode="anchored",
            init_mode="around_best",
            init_scales={"tstar_frac": 0.1, "tdust_frac": 0.1, "log10_tau": 0.3, "log10_a": 0.3},
        ),
        "mixture": dict(
            posterior_mode="mixture",
            mix_weight=args.mix_weight,
            init_mode="hybrid",
            init_scales={"tstar_frac": 0.1, "tdust_frac": 0.1, "log10_tau": 0.3, "log10_a": 0.3},
        ),
    }

    outdir = Path(default_outdir) / oid
    outdir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        print(f"\n=== Running mode: {mode} ===", flush=True)
        
        run_id_suffix = f"_{args.run_id}" if args.run_id else ""
        run_tag = f"{oid}_{('tmpl' if template_mode else 'bb')}_thick{str(shell_thickness).replace('.', '_')}_{mode}_seed{args.seed}{run_id_suffix}"

        results = run_mcmc_for_sed(
            sed=sed,
            grid_df=df,
            dusty_file_dir=dusty_file_dir,
            workdir=workdir,
            dust_type=dust_type,
            shell_thickness=shell_thickness,
            template_path=(args.template_path if template_mode else None),
            template_tag=args.template_tag if template_mode else None,
            template_kind=args.template_kind,
            spectrum_z=args.spectrum_z,
            tstar_dummy=float(args.tstar_dummy),
            nwalkers=args.nwalkers,
            nsteps=args.nsteps,
            burn_in=args.burnin,
            n_cores=args.ncores,
            mp_prefer=args.mp,
            random_seed=args.seed,
            progress_every=args.progress_every,
            cache_dir=args.cache_dir,
            cache_ndigits=args.cache_ndigits,
            cache_max=args.cache_max,
            use_tmp=not args.no_tmp,
            run_tag=run_tag,
            evaluator_mode=args.evaluator_mode,
            emulator_path=args.emulator_path,
            use_weights=use_weights,
            sample_log10a=args.sample_log10a,
            log10_a_range=args.log10a_range,
            err_floor_frac=args.err_floor_frac,
            equal_weights=args.equal_weights,
            quadratic_weights=args.quadratic_weights,
            spectrum=spectrum,
            spectrum_err_floor_frac=args.spectrum_err_floor_frac,
            **mode_kwargs[mode],
        )

        suffix = f"_seed{args.seed}"
        thick_str = str(shell_thickness).replace('.', '_')

        a_tag = "anl" if not args.sample_log10a else "smpl"

        if not use_weights:
            outname = (
                f"mcmc_{oid}_{('template' if template_mode else 'bb')}"
                f"_thick{thick_str}_{mode}_{a_tag}_no_weights_seed{args.seed}"
                f"{run_id_suffix}.npz"
            )
        else:
            outname = (
                f"mcmc_{oid}_{('template' if template_mode else 'bb')}"
                f"_thick{thick_str}_{mode}_{a_tag}_seed{args.seed}"
                f"{run_id_suffix}.npz"
            )

        outpath = outdir / outname

        np.savez(
            outpath,
            samples=results["samples"],
            log_prob=results["log_prob"],
            chain=results["chain"],
            log_prob_chain=results["log_prob_chain"],
            acceptance_fraction=results["acceptance_fraction"],
            autocorr_time=results["autocorr_time"],
            tstar=results["tstar"],
            tdust=results["tdust"],
            tau=results["tau"],
            a=results["a"],
            grid_best=results["grid_best"],
            prior_config=results["prior_config"],
            log10_a_samples=results["log10_a_samples"],
            analytic_log10a=results["analytic_log10a"],
            seed=args.seed,
            mode=mode,
            template_mode=results.get("template_mode", False),
            template_tag=results.get("template_tag", ""),
            shell_thickness=shell_thickness,
            evaluator_mode=results.get("evaluator_mode", "dusty"),
            err_floor_frac=args.err_floor_frac,
            equal_weights=args.equal_weights,
            quadratic_weights=args.quadratic_weights,
            spectrum_path=str(args.spectrum_path) if args.spectrum_path else "",
            spectrum_err_floor_frac=args.spectrum_err_floor_frac,
            group_weight_scales=str(results.get("group_weight_scales")),
        )
        
        print(f"Saved results to: {outpath}")
    
    print("Done.")

if __name__ == "__main__":
    main()