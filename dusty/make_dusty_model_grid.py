import os
# keep single-threaded for numerical libraries to avoid oversubscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import argparse
from pathlib import Path
import numpy as np
import csv
import shutil
import stat
import subprocess

# pydusty imports
from pydusty.dusty import DustyParameters, Dusty, Dusty_Alumina_SilDL
from pydusty.parameters import Parameter
from pydusty.utils import getLogger

# parallel processing
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools as it

def run_single_model(job):
    """Run a single DUSTY model in its own directory and write sed.dat with given parameters."""
    (tstarval, tdustval, tauval,
     args, dust_type, blackbody, shell_thickness,
     tstarmin, tstarmax, custom_grain_distribution,
     tau_wav_micron, base_workdir) = job
    
    base_workdir = Path(base_workdir).resolve()

    leaf = f"Tstar_{int(tstarval)}_Tdust_{int(tdustval)}_tau_{tauval:g}".replace('.', '_')
    run_dir = (base_workdir / leaf).resolve() 
    run_dir.mkdir(parents=True, exist_ok=True)

    # ensure a 'dusty' binary is present in this run dir (symlink to the source)
    src = Path(args.dusty_file_dir).resolve() / "dusty"
    dst = run_dir / "dusty"

    if not src.exists():
        return dict(tstar=tstarval, tdust=tdustval, tau=tauval,
                    outpath=None, r1=None, ierror=1,
                    error=f"DUSTY binary not found at {src}", cached=False)

    if not dst.exists():
        shutil.copy2(src, dst)
        # ensure it’s executable
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if not os.access(dst, os.X_OK):
        return dict(tstar=tstarval, tdust=tdustval, tau=tauval,
                    outpath=None, r1=None, ierror=1,
                    error=f"DUSTY binary at {dst} is not executable", cached=False)

    # Make sure PATH includes the run dir (some wrappers call 'dusty' by name)
    os.environ["PATH"] = f"{str(run_dir)}:{str(Path(args.dusty_file_dir).resolve())}:{os.environ.get('PATH','')}"

    outpath = run_dir / "sed.dat"
    if outpath.exists():
        # already done
        return dict(tstar=tstarval, tdust=tdustval, tau=tauval, outpath=outpath, r1=None, ierror=0, cached=True)
    
    # Build DUSTY parameters
    tstar = Parameter(name="tstar", value=float(tstarval), is_variable=False)
    tdust = Parameter(name="tdust", value=float(tdustval), is_variable=True)
    tau = Parameter(name="tau", value=float(tauval), is_variable=False)

    dusty_parameters = DustyParameters(
        tstar=tstar,
        tdust=tdust,
        tau=tau,
        blackbody=blackbody,
        shell_thickness=shell_thickness,
        dust_type=dust_type,
        tstarmin=tstarmin,
        tstarmax=tstarmax,
        custom_grain_distribution=custom_grain_distribution,
        tau_wavelength_microns=tau_wav_micron,
    )

    runner = Dusty(
        parameters=dusty_parameters,
        dusty_working_directory=str(run_dir),
        dusty_file_directory=Path(args.dusty_file_dir).resolve(),
    )
    
    # Run DUSTY
    prev_cwd = os.getcwd()
    try:
        os.chdir(str(run_dir))
        os.environ["PATH"] = f"{str(run_dir)}:{str(Path(args.dusty_file_dir).resolve())}:{os.environ.get('PATH','')}"
        print(f"[debug] CWD -> {os.getcwd()}")
        print("generating input...")
        runner.generate_input()
        print("running dusty...")
        runner.run()
        print("ran dusty, getting results...")
        lam, flx, npt, r1, ierror = runner.get_results()
    except Exception as e:
        os.chdir(prev_cwd)
        print("E:", e)
        return dict(tstar=tstarval, tdust=tdustval, tau=tauval, outpath=None, ierror=1, error=str(e), cached=False)

    os.chdir(prev_cwd)

    # Save output
    if ierror == 0:
        outpath = run_dir / "sed.dat"
        with outpath.open('w') as f:
            f.write(f"# {r1}\n")
            f.write("lam, flux\n")
            for ind in range(len(lam)):
                f.write(f"{lam[ind]}, {flx[ind]}\n")
    
    return dict(tstar=tstarval, tdust=tdustval, tau=tauval, outpath=outpath, r1=r1, ierror=ierror, cached=False)

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tau_wav_micron", type=float, default=0.55,
        help="Wavelength (in microns) at which tau is specified (use 0.55 micron for v band)."
    )
    parser.add_argument(
        "--thick", type=float, default=2.0,
        help="Shell thickness (R_out / R_in)."
    )
    parser.add_argument("--dtype", choices=['graphite', 'silicate',
                                            'amorphous_carbon', 'silicate_carbide'],
                        default='silicate',
                        help="Dust type to use.")
    # parser.add_argument(
    #     "--al", type=float, default=0.0,
    #     help="Aluminum abundance, Silicon abundance is 1-al."
    # )
    # parser.add_argument(
    #     "--al_type", type=str, default="compact",
    #     choices=["compact", "porous"],
    #     help="Alumina porosity type."
    # )
    parser.add_argument(
        "workdir", type=str,
        help="Directory in which to run/store DUSTY outputs."
    )
    parser.add_argument(
        "--dusty_file_dir", type=str, default="/Users/ana/Documents/dusty/releaseV2",
        help="Directory with DUSTY code files."
    )
    parser.add_argument(
        "--loglevel", type=str, default="DEBUG",
        help="Logging level."
    )
    parser.add_argument(
        "--logfile", type=str, default=None,
        help="Log file path."
    )

    ### Additional arguments for grid parameters
    parser.add_argument("--tstar_list", type=str, default=None,
                    help="Comma-separated T* values in K, e.g. '4000,4500'")
    parser.add_argument("--tdust_list", type=str, default=None,
                        help="Comma-separated Tdust values in K, e.g. '900,1000'")
    parser.add_argument("--tau_list", type=str, default=None,
                        help="Comma-separated tauV values, e.g. '0.03,0.1,0.3'")

    args = parser.parse_args()

    logger = getLogger(args.loglevel, args.logfile)

    # -----------------------------
    # Physically motivated grids
    # -----------------------------

    # Stellar temperature (K): cool photosphere range appropriate for your source
    tstar_values = [3500, 4000, 4500, 5000, 5500, 6000]

    # Inner dust temperature (K): warm silicate near sublimation
    tdust_values = [800, 900, 1000, 1100, 1200, 1300]

    # Optical depth at V-band: thin -> moderate (avoid extreme thickness with only 4 bands)
    # (log-spaced with a bit more density at thin end)
    tau_values = np.r_[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0]

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

    # -----------------------------
    # Build parameter objects
    # -----------------------------
    
    blackbody = Parameter(name="blackbody", value=True)
    shell_thickness = Parameter(name="shell_thickness", value=args.thick)

    # Dust label: pure silicate (DL) with optional alumina fraction
    # dust_label = f'si_{(1 - args.al)}_al_{args.al}_{args.al_type}_tau_{args.tau_wav_micron}um'
    dust_label = f'{args.dtype}_tau_{args.tau_wav_micron}um'
    dust_type = Parameter(name="dust_type", value=args.dtype)

    # Bounds metadata
    tstarmin = Parameter(name="tstarmin", value=3000)
    tstarmax = Parameter(name="tstarmax", value=10000)

    custom_grain_distribution = Parameter(name="custom_grain_distribution", value=False)
    tau_wav_micron = Parameter(name="tau_wav", value=args.tau_wav_micron, is_variable=False)
    # al_abundance = Parameter(name="al", value=args.al, is_variable=False)

    # -----------------------------
    # Paths (no chdir)
    # -----------------------------

    workdir = (Path(args.workdir) / f'{dust_label}_thick_{shell_thickness.value}').resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    dusty_dir_abs = Path(args.dusty_file_dir).resolve()
    dusty_bin = (dusty_dir_abs / "dusty")
    if not dusty_bin.exists():
        raise FileNotFoundError(f"Missing DUSTY binary at {dusty_bin}. Build it or fix --dusty_file_dir.")
    os.environ["PATH"] = f"{str(dusty_dir_abs)}:{os.environ.get('PATH','')}"


    # -----------------------------
    # Parallel grid execution
    # -----------------------------

    jobs = [
        (t, d, tv, 
         args, dust_type, blackbody, shell_thickness,
         tstarmin, tstarmax, custom_grain_distribution,
         tau_wav_micron, workdir)
         for t, d, tv in it.product(tstar_values, tdust_values, tau_values)
    ]

    max_workers = min(os.cpu_count() or 2, 4)

    results_summary = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_model, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results_summary.append(res)
            status = "OK" if res.get("ierror", 1) == 0 else "ERROR"
            logger.info(f"{status}  T*={res['tstar']} Td={res['tdust']} tau={res['tau']}  -> {res.get('outpath')}")
    
    # Write summary CSV
    idx_path = Path(workdir) / "model_grid_summary.csv"
    with idx_path.open('w', newline='') as csvfile:
        fieldnames = ["tstar", "tdust", "tau", "outpath", "r1", "ierror", "cached", "error"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results_summary:
            writer.writerow({k: row.get(k) for k in fieldnames})
    logger.info(f"Wrote model grid summary to {idx_path}")

if __name__ == "__main__":
    main()