# Run with:
# python -m lltypeiip.inference.run_sed_mcmc <OID>

import argparse
import os
import numpy as np
from pathlib import Path
import pickle
import pandas as pd

from lltypeiip.dusty.scaling import fit_template_grid_to_sed


from ..sed import build_multi_epoch_seds_from_tail
from .mcmc import run_mcmc_for_sed
from ..config import config, PROJECT_ROOT
from ..photometry import get_wise_lc_data, get_ztf_lc_data, convert_ZTF_mag_mJy
from ..dusty import fit_grid_to_sed, NugentIIPSeries, DustyRunner

from alerce.core import Alerce

default_workdir = config.dusty.workdir_mcmc_dir
default_outdir  = getattr(config.paths, "mcmc_results_dir", str(PROJECT_ROOT / "mcmc_results"))
default_griddir = getattr(config.paths, "dusty_grid_dir", str(PROJECT_ROOT / "dusty_runs/silicate_tau_0.55um_fixed_thick_grid"))
default_cache_dir = config.dusty.npz_cache_dir 

def parse_args():

    p = argparse.ArgumentParser(description="Run DUSTY MCMC for SN SED")

    # required
    p.add_argument("oid", type=str, help="ZTF object ID (e.g. ZTF22abtspsw)")

    # template mode
    p.add_argument("--template-path", type=str, default=None,
                   help="If set: use Spectrum=5 template mode.")
    p.add_argument("--template-tag", type=str, default="nugent_iip")
    p.add_argument("--template-grid-csv", type=str, default=None,
                   help="CSV from run_template_grid.py (full rows), used to pick best row for initialization.")
    p.add_argument("--tstar-dummy", type=float, default=6000.0)

    # optionally load one saved sed pickle instead of rebuilding
    p.add_argument("--sed-pkl", type=str, default=None,
                   help="Optional: path to a saved tail_sed pickle. Can be plain SED dict or payload with key 'sed'.")

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

    # mixture
    p.add_argument("--mix-weight", type=float, default=0.3,
                   help="Gaussian weight for mixture prior")

    p.add_argument("--workdir", type=str, default=default_workdir)
    p.add_argument("--grid-dir", type=str, default=default_griddir)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mp", choices=["spawn", "fork"], default="spawn")

    # cache
    p.add_argument("--cache-dir", type=str, default=default_cache_dir,
               help="On-disk cache for DUSTY results (npz). Default: <workdir>/dusty_npz_cache")
    p.add_argument("--cache-ndigits", type=int, default=4,
                help="Rounding digits used for tau/thickness cache keys.")
    p.add_argument("--cache-max", type=int, default=5000,
                help="Max number of models kept in memory cache per process.")
    p.add_argument("--no-tmp", action="store_true", help="Disable temp dirs for DUSTY runs")

    # shell thickness
    p.add_argument("--shell-thickness", type=float, default=2.0,
                   help="Shell thickness (Y_out/Y_in). Default: 2.0")

    return p.parse_args()

def _unwrap_sed(obj):
    if isinstance(obj, dict) and "sed" in obj and isinstance(obj["sed"], dict):
        sed = obj["sed"]
        if "phase_days" not in sed and "phase" in obj:
            sed["phase_days"] = obj["phase"]
        return sed
    return obj

def main():
    
    args = parse_args()

    oid = args.oid

    template_mode = (args.template_path is not None)

    print(f"Running MCMC for OID: {oid}")
    print(f"Posterior mode: {args.mode}")

    dusty_file_dir = config.dusty.dusty_file_dir
    workdir = args.workdir
    grid_dir = args.grid_dir

    os.makedirs(workdir, exist_ok=True)

    print(f"DUSTY binary dir: {dusty_file_dir}")
    print(f"Working directory: {workdir}")
    print(f"Grid directory: {grid_dir}")

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

    # grid fitting
    print("Running grid fit...")

    if template_mode:
        if args.template_grid_csv is None:
            raise RuntimeError("Template mode requires --template-grid-csv from run_template_grid.py")

        template = NugentIIPSeries(str(Path(args.template_path).resolve()))

        runner = DustyRunner(
            base_workdir=workdir,
            dusty_file_dir=dusty_file_dir,
            dust_type="silicate", 
            shell_thickness=args.shell_thickness,
            cache_dir=args.cache_dir,
            cache_ndigits=args.cache_ndigits,
            cache_max=args.cache_max,
            use_tmp=not args.no_tmp,
        )

        df = fit_template_grid_to_sed(
            template_grid_csv=args.template_grid_csv,
            sed=sed,
            runner=runner,
            template=template,
            template_tag=args.template_tag,
            y_mode="Flam",
            use_weights=True,
        )


        df_all = pd.read_csv(Path(args.template_grid_csv).resolve())
        df_filtered = df_all[
            (df_all["oid"].astype(str) == str(oid)) &
            (df_all["shell_thickness"] == args.shell_thickness)
        ].copy()
        if df_filtered.empty:
            raise RuntimeError(f"No rows for oid={oid} found in {args.template_grid_csv}")

        df_filtered = df_filtered.sort_values("chi2_red")
        df = df_filtered  # pass into run_mcmc_for_sed, it uses df.iloc[0] as best
        shell_thickness = args.shell_thickness
        dust_type = str(df.iloc[0].get("dust_type", "silicate")) if "dust_type" in df.columns else "silicate"
    else:
        df = fit_grid_to_sed(
            grid_dir,
            sed,
            y_mode="Flam",
            use_weights=True
        )
        shell_thickness = args.shell_thickness
        dust_type = "silicate"


    # MCMC 
    print("Starting MCMC...")
    print(f"mp_prefer={args.mp}  ncores={args.ncores}  nwalkers={args.nwalkers}", flush=True)

    modes = [args.mode] if not args.sweep else ["data", "anchored", "mixture"]


    mode_kwargs = {
        "data": dict(
            posterior_mode="data",
            init_mode="hybrid",
            init_scales={"tstar_frac": 0.3, "tdust_frac": 0.3, "log10_tau": 1.0, "log10_a": 1.0},
        ),
        "anchored": dict(
            posterior_mode="anchored",
            init_mode="around_best",
            # keep your default anchored widths in run_mcmc_for_sed, or pass explicitly if you want
        ),
        "mixture": dict(
            posterior_mode="mixture",
            mix_weight=args.mix_weight,
            init_mode="hybrid",
            init_scales={"tstar_frac": 0.3, "tdust_frac": 0.3, "log10_tau": 1.0, "log10_a": 1.0},
        ),
    }

    outdir = Path(default_outdir) / oid
    outdir.mkdir(parents=True, exist_ok=True)

    for mode in modes:
        print(f"\n=== Running mode: {mode} ===", flush=True)
        run_tag = f"{oid}_{('tmpl' if template_mode else 'bb')}_{mode}_seed{args.seed}"
        results = run_mcmc_for_sed(
            sed=sed,
            grid_df=df,
            dusty_file_dir=dusty_file_dir,
            workdir=workdir,
            dust_type=dust_type,
            shell_thickness=shell_thickness,
            template_path=(args.template_path if template_mode else None),
            template_tag=args.template_tag,
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
            **mode_kwargs[mode],
        )

        suffix = "" if args.seed == 42 else f"_seed{args.seed}"
        thick_str = str(shell_thickness).replace('.', '_')
        outname = f"mcmc_{oid}_{('template' if template_mode else 'bb')}_thick{thick_str}_{mode}{suffix}.npz"
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
            seed=args.seed,
            mode=mode,
            template_mode=results.get("template_mode", False),
            template_tag=results.get("template_tag", ""),
        )

        print(f"Saved results to: {outpath}")
    print("Done.")

if __name__ == "__main__":
    main()