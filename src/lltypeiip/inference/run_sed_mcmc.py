# Run with:
# python -m lltypeiip.inference.run_sed_mcmc <OID>

import argparse
import os
import numpy as np
from pathlib import Path

from ..sed import build_multi_epoch_seds_from_tail
from .mcmc import run_mcmc_for_sed
from ..config import config, PROJECT_ROOT
from ..photometry import get_wise_lc_data, get_ztf_lc_data, convert_ZTF_mag_mJy
from ..dusty import fit_grid_to_sed

from alerce.core import Alerce

default_workdir = config.dusty.workdir_mcmc_dir
default_outdir  = getattr(config.paths, "mcmc_results_dir", str(PROJECT_ROOT / "mcmc_results"))
default_griddir = getattr(config.paths, "dusty_grid_dir", str(PROJECT_ROOT / "dusty_runs/silicate_tau_0.55um_fixed_thick_grid"))

def parse_args():

    p = argparse.ArgumentParser(description="Run DUSTY MCMC for SN SED")

    # required
    p.add_argument("oid", type=str, help="ZTF object ID (e.g. ZTF22abtspsw)")

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

    return p.parse_args()

def main():
    
    args = parse_args()

    oid = args.oid
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
        require_wise_detection=True
    )

    if len(seds) == 0:
        raise RuntimeError("No valid SEDs constructed (WISE detection required).")

    sed = seds[0]
    print(f"Using SED epoch at MJD ~ {np.nanmean(sed['mjd']):.1f}")

    # grid fitting
    print("Running grid fit...")
    df = fit_grid_to_sed(
        grid_dir,
        sed,
        y_mode="Flam",
        use_weights=True
    )

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
        results = run_mcmc_for_sed(
            sed=sed,
            grid_df=df,
            dusty_file_dir=dusty_file_dir,
            workdir=workdir,
            nwalkers=args.nwalkers,
            nsteps=args.nsteps,
            burn_in=args.burnin,
            n_cores=args.ncores,
            mp_prefer=args.mp,
            random_seed=args.seed,
            progress_every=args.progress_every,
            **mode_kwargs[mode],
        )

        suffix = "" if args.seed == 42 else f"_seed{args.seed}"
        outname = f"mcmc_{oid}_{mode}{suffix}.npz"
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
        )

        print(f"Saved results to: {outpath}")
    print("Done.")

if __name__ == "__main__":
    main()