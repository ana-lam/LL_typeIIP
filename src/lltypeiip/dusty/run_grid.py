# run as such from home directory
# python -m lltypeiip.dusty.run_grid \
# --thick_list "2.0" \
# --n_workers 4 "dusty_runs/blackbody_grids"
import os
# keep single-threaded for numerical libraries to avoid oversubscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from ..config import config
from ..dusty.runner import DustyRunner

# pydusty imports
from pydusty.dusty import DustyParameters, Dusty, Dusty_Alumina_SilDL
from pydusty.parameters import Parameter
from pydusty.utils import getLogger

# parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools as it

# worker globals
_G = {
    "runner": None,
    "runner_sig": None,
}


def make_leaf_and_key(tstar, tdust, tau, thick, ndigits=6):
    """
    Canonical naming + canonical float key.
    """

    tstar_i = int(round(float(tstar)))
    tdust_i = int(round(float(tdust)))
    tau_f   = round(float(tau), ndigits)
    thick_f = round(float(thick), ndigits)

    leaf = (f"Tstar_{tstar_i}_Tdust_{tdust_i}_"
            f"tau_{tau_f:.{ndigits}g}_thick_{thick_f:.{ndigits}g}").replace('.', '_')

    key = (tstar_i, tdust_i, tau_f, thick_f)

    return leaf, key

def get_runner(runner_sig, runner_kwargs):
    """Get or create a global DustyRunner instance."""
    global _G
    if (_G["runner"] is None) or (_G["runner_sig"] != runner_sig):
        _G["runner"] = DustyRunner(**runner_kwargs)  
        _G["runner_sig"] = runner_sig
    return _G["runner"]

def run_single_model(job):
    """Run a single DUSTY model in its own directory and write sed.dat with given parameters."""
    (tstarval, tdustval, tauval, shell_thick_val,
    runner_kwargs, runner_sig, ndigits,
    base_workdir, write_sed_dat) = job
    
    base_workdir = Path(base_workdir).resolve()

    # get runner (cached per worker)
    runner = get_runner(runner_sig, runner_kwargs)

    leaf, key = make_leaf_and_key(tstarval, tdustval, tauval, shell_thick_val, ndigits=ndigits)
    
    lam_um, lamFlam, r1 = runner.evaluate_model(
        tstar=tstarval,
        tdust=tdustval,
        tau=tauval,
        shell_thickness=shell_thick_val,
        template=None, # blackbody
        phase_days=None,
        template_tag=None
    )

    ckey = runner._canonical_key(tstarval, tdustval, tauval,
                                 shell_thick_val)
    npz_path = runner._disk_cache_path(ckey)

    if lam_um is None or lamFlam is None:
        return dict(tstar=tstarval, tdust=tdustval, tau=tauval, 
                    shell_thickness=shell_thick_val,
                    outpath=None, npz_path=str(npz_path) if npz_path else None,
                    r1=np.nan, ierror=1, error="DUSTY failed", cached=False)

    # Optionally save sed.dat
    outpath=None
    if write_sed_dat:
        run_dir = base_workdir / leaf
        run_dir.mkdir(parents=True, exist_ok=True)
        outpath= run_dir / "sed.dat"

        try:
            with outpath.open('w') as f:
                f.write(f"# {r1}\n")
                f.write("lam, flux\n")
                for lam_val, flux_val in zip(lam_um, lamFlam):
                    f.write(f"{lam_val}, {flux_val}\n")
        except Exception as e:
            print(f"Error writing sed.dat: {e}")
            outpath=None
    
    return dict(
        tstar=tstarval,
        tdust=tdustval,
        tau=tauval,
        shell_thickness=shell_thick_val,
        outpath=outpath if outpath else None,
        npz_path=str(npz_path) if npz_path else None,
        r1=r1,
        ierror=0,
        error=None,
        cached=False
    )

def main():
    
    # Use config from lltypeiip.config
    default_dusty_file_dir = config.dusty.dusty_file_dir
    default_workdir = config.dusty.workdir + "/blackbody_grids" # will actually define in command run
    default_cache_dir = config.dusty.npz_cache_dir
    
    parser = argparse.ArgumentParser(
        description="Generate DUSTY grid models"
    )

    parser.add_argument("--tau_wav_micron", type=float, default=0.55,
                        help="Wavelength (in microns) at which tau is specified (use 0.55 micron for v band).")
    
    parser.add_argument("--dust_type", choices=['graphite', 'silicate',
                                            'amorphous_carbon', 'silicate_carbide'],
                        default='silicate',
                        help="Dust type to use.")
    
    parser.add_argument("workdir", type=str,
                        help="Directory in which to run/store DUSTY outputs.")
    
    parser.add_argument("--dusty_file_dir", type=str, default=default_dusty_file_dir,
                        help="Directory with DUSTY code files.")
    
    parser.add_argument("--loglevel", type=str, default="INFO",
                        help="Logging level.")
    
    parser.add_argument("--logfile", type=str, default=None,
                        help="Log file path.")

    ### Additional arguments for grid parameters
    parser.add_argument("--tstar_list", type=str, default=None,
                        help="Comma-separated T* values in K, e.g. '4000,4500'")
    parser.add_argument("--tdust_list", type=str, default=None,
                        help="Comma-separated Tdust values in K, e.g. '900,1000'")
    parser.add_argument("--tau_list", type=str, default=None,
                        help="Comma-separated tauV values, e.g. '0.03,0.1,0.3'")
    parser.add_argument("--thick_list", type=str, default="2.0",
                        help="Comma-separated shell thickness R_out/R_in values, e.g. '1.2,1.5,2,3,5'.")


    parser.add_argument("--write_sed_dat", action="store_true",
                        help="Write sed.dat files.")

    # caching args
    parser.add_argument("--cache_dir", type=str, default=default_cache_dir,
                        help="Directory for .npz cache. Default under workdir.")
    parser.add_argument("--cache_ndigits", type=int, default=4)
    parser.add_argument("--cache_max", type=int, default=5000)
    parser.add_argument("--use_tmp", action="store_true",
                        help="Use temporary directories (auto-cleanup).")


    parser.add_argument("--force_rerun", action="store_true",
                        help="Force re-running all models, ignoring any cached results.")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="Number of parallel workers to use.")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Optional output CSV file for summary of results.")

    args = parser.parse_args()

    logger = getLogger(args.loglevel, args.logfile)

    # -----------------------------
    # Physically motivated grids
    # -----------------------------

    # Stellar temperature (K): cool photosphere range appropriate for your source
    tstar_values = [4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000,
                    10500, 11000, 11500, 12000, 12500, 13000, 13500, 14000, 14500, 15000]

    # Inner dust temperature (K): warm silicate near sublimation
    tdust_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    # Optical depth at V-band: thin -> moderate (avoid extreme thickness with only 4 bands)
    # (log-spaced with a bit more density at thin end)
    tau_values = np.r_[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # Thickness (R_out / R_in)
    shell_thickness_values = [2.0]

    # Overrides from CLI if given
    def _parse_list(s):
        # accepts "4000,4500" or "4e3,5e3"
        return [float(x) for x in s.split(",") if x.strip()]

    if args.tstar_list: 
        tstar_values = _parse_list(args.tstar_list)
    if args.tdust_list: 
        tdust_values = _parse_list(args.tdust_list)
    if args.tau_list:   
        tau_values = _parse_list(args.tau_list)
    if args.thick_list:
        shell_thickness_values = _parse_list(args.thick_list)

    # -----------------------------
    # Build parameter objects
    # -----------------------------
    
    blackbody = Parameter(name="blackbody", value=True)

    # Dust label: pure silicate (DL) with optional alumina fraction
    # dust_label = f'si_{(1 - args.al)}_al_{args.al}_{args.al_type}_tau_{args.tau_wav_micron}um'
    dust_label = f'{args.dust_type}_tau_{args.tau_wav_micron}um'
    dust_type = Parameter(name="dust_type", value=args.dust_type)

    # Bounds metadata
    tstarmin = Parameter(name="tstarmin", value=1000)
    tstarmax = Parameter(name="tstarmax", value=15000)

    custom_grain_distribution = Parameter(name="custom_grain_distribution", value=False)
    tau_wav_micron = Parameter(name="tau_wav", value=args.tau_wav_micron, is_variable=False)

    # -----------------------------
    # Paths (no chdir)
    # -----------------------------

    # Resolve workdir as absolute path
    # If args.workdir is relative, make it relative to project root (parent of src/)
    workdir_path = Path(args.workdir)
    if not workdir_path.is_absolute():
        # Find project root by going up from the script location
        script_dir = Path(__file__).parent.resolve()  # lltypeiip/dusty/
        project_root = script_dir.parent.parent.parent  # LL_typeIIP/
        workdir_path = (project_root / args.workdir).resolve()
    
    workdir = (workdir_path / f'{dust_label}_thick_{str(shell_thickness_values[0]).replace(".", "_")}').resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    dusty_dir_abs = Path(args.dusty_file_dir).resolve()
    dusty_bin = (dusty_dir_abs / "dusty")
    if not dusty_bin.exists():
        raise FileNotFoundError(f"Missing DUSTY binary at {dusty_bin}. Build it or fix --dusty_file_dir.")
    # os.environ["PATH"] = f"{str(dusty_dir_abs)}:{os.environ.get('PATH','')}"

    if args.out_csv is None:
        out_csv = workdir / f"grid_summary_blackbody_thick_{str(shell_thickness_values[0]).replace('.', '_')}.csv"
    else:
        out_csv = Path(args.out_csv).resolve()

    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(config.dusty.npz_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------
    # Runner kwargs and signature
    # ----------------------------- 
    runner_kwargs = dict(
        base_workdir=str(workdir),
        dusty_file_dir=str(dusty_dir_abs),
        dust_type=args.dust_type,
        shell_thickness=float(shell_thickness_values[0]) if len(shell_thickness_values) == 1 else 2.0,
        tau_wavelength_microns=float(args.tau_wav_micron),
        cache_dir=str(cache_dir),
        cache_ndigits=(args.cache_ndigits),
        cache_max=(args.cache_max),
        use_tmp=bool(args.use_tmp),
        run_tag=f"grid_{dust_label}",
        blackbody=True,
        tstarmin=1000.,
        tstarmax=15000.,
    )
    runner_sig = (
        runner_kwargs["base_workdir"],
        runner_kwargs["dusty_file_dir"],
        runner_kwargs["dust_type"],
        runner_kwargs["tau_wavelength_microns"],
        runner_kwargs["cache_dir"],
        runner_kwargs["cache_ndigits"],
        runner_kwargs["cache_max"],
        runner_kwargs["use_tmp"],
        runner_kwargs["run_tag"]
    )

    # -----------------------------
    # Check cache
    # -----------------------------
    completed = set()

    if not args.force_rerun:
        if out_csv.exists():
            prev = pd.read_csv(out_csv)
            prev_ok = prev[(prev["ierror"] == 0) & (prev["npz_path"].notna() | prev["outpath"].notna())]
            for _, r in prev_ok.iterrows():
                _, k = make_leaf_and_key(r["tstar"], r["tdust"], r["tau"], r["shell_thickness"], args.cache_ndigits)
                completed.add(k)
            logger.info(f"Found {len(completed)} completed models to skip.")

    # -----------------------------
    # Parallel grid execution
    # -----------------------------

    jobs = []
    skipped = 0

    for t, d, tv, thick in it.product(tstar_values, tdust_values, tau_values, shell_thickness_values):
        leaf, key = make_leaf_and_key(t, d, tv, thick, ndigits=args.cache_ndigits)
        if key in completed:
            skipped += 1
            continue

        jobs.append(
        (t, d, tv, thick,
         runner_kwargs, runner_sig, args.cache_ndigits,
         workdir, args.write_sed_dat)
        )

    logger.info(f"Skipping {skipped} cached models, running {len(jobs)} new ones.")

    if len(jobs) == 0:
        logger.info("No models to run (all cached).")
        return

    max_workers = min(os.cpu_count() or 2, int(args.n_workers))

    results_summary = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_model, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results_summary.append(res)
            status = "OK" if res.get("ierror", 1) == 0 else "ERROR"
            logger.info(
                f"{status}  T*={res['tstar']} Td={res['tdust']} tau={res['tau']} "
                f"thick={res['shell_thickness']}  -> npz={res.get('npz_path', 'N/A')}"
                )
    
    # Write summary CSV
    if results_summary:
        # load existing results if any
        if not args.force_rerun and out_csv.exists():
            existing = pd.read_csv(out_csv)
            new_df = pd.DataFrame(results_summary)
            df = pd.concat([existing, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(results_summary)

        df = df.sort_values(["tstar", "tdust", "tau", "shell_thickness"])
        df.to_csv(out_csv, index=False)
        logger.info(f"Wrote model grid summary to {out_csv}")
    else:
        logger.warning("No results to write.")

if __name__ == "__main__":
    main()