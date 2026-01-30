import os
# keep single-threaded for numerical libraries to avoid oversubscription
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import argparse
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

from ..config import config
from ..dusty.runner import DustyRunner
from ..dusty.custom_spectrum import NugentIIPSeries

from pydusty.utils import getLogger
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools as it


# ---------------------------
# Globals per worker process
# ---------------------------
_G = {
    "template": None,
    "template_path": None,
    "runner": None,
    "runner_sig": None,
}


def _parse_list(s):
    """Parse comma-separated list of floats."""
    return [float(x) for x in s.split(",") if x.strip()]


def make_leaf_and_key(tstar, tdust, tau, thick, phase, template_tag, ndigits=4):
    """
    Canonical naming + canonical key for template-based models.
    Mirrors run_grid.py's make_leaf_and_key but adds phase and template_tag.
    """
    tstar_i = int(round(float(tstar)))
    tdust_i = int(round(float(tdust)))
    tau_f = round(float(tau), ndigits)
    thick_f = round(float(thick), ndigits)
    phase_f = round(float(phase), 2)
    tag = str(template_tag)
    
    leaf = (f"Tstar_{tstar_i}_Tdust_{tdust_i}_"
            f"tau_{tau_f:.{ndigits}g}_thick_{thick_f:.{ndigits}g}_"
            f"{tag}_phase_{phase_f:.2f}").replace('.', '_')
    
    key = (tstar_i, tdust_i, tau_f, thick_f, phase_f, tag)
    
    return leaf, key


def _get_template(template_path: str):
    """
    Load Nugent template once per worker process.
    """
    global _G
    template_path = str(Path(template_path).resolve())
    if (_G["template"] is None) or (_G["template_path"] != template_path):
        _G["template"] = NugentIIPSeries(template_path)
        _G["template_path"] = template_path
    return _G["template"]


def _get_runner(runner_sig: tuple, runner_kwargs: dict):
    """
    Create DustyRunner once per worker process.
    """
    global _G
    if (_G["runner"] is None) or (_G["runner_sig"] != runner_sig):
        _G["runner"] = DustyRunner(**runner_kwargs)
        _G["runner_sig"] = runner_sig
    return _G["runner"]


def run_single_model(job):
    """
    Run a single (tstar, tdust, tau, thick, phase) model in its own directory and 
    write sed.dat with given parameters.
    """
    (tstarval, tdustval, tauval, thick_val, phase_days, oid,
     template_path, template_tag,
     runner_kwargs, runner_sig, ndigits) = job
    
    # Get template and runner (cached per worker)
    template = _get_template(template_path)
    runner = _get_runner(runner_sig, runner_kwargs)
    
    # Canonical key for this model
    leaf, key = make_leaf_and_key(
        tstarval, tdustval, tauval, thick_val, phase_days, template_tag, ndigits
    )
    
    # Run DUSTY model
    lam_um, lamFlam, r1 = runner.evaluate_model(
        tstar=float(tstarval),
        tdust=float(tdustval),
        tau=float(tauval),
        shell_thickness=float(thick_val),
        template=template,
        phase_days=float(phase_days),
        template_tag=str(template_tag),
    )

    ckey = runner._canonical_key(
        tstarval, tdustval, tauval, thick_val,
        phase_days=phase_days,
        template_tag=template_tag
    )
    npz_path = runner._disk_cache_path(ckey)
    
    if lam_um is None or lamFlam is None:
        return dict(
            oid=str(oid),
            phase_days=float(phase_days),
            template_tag=str(template_tag),
            tstar_dummy=float(tstarval),
            tdust=float(tdustval),
            tau=float(tauval),
            shell_thickness=float(thick_val),
            r1=np.nan,
            npz_path=str(npz_path) if npz_path else None,
            ierror=1,
            error="DUSTY failed",
            cached=False
        )

    return dict(
        oid=str(oid),
        phase_days=float(phase_days),
        template_tag=str(template_tag),
        tstar_dummy=float(tstarval),
        tdust=float(tdustval),
        tau=float(tauval),
        shell_thickness=float(thick_val),
        r1=float(r1),
        npz_path=str(npz_path) if npz_path else None,
        ierror=0,
        error=None,
        cached=False
    )


def main():
    default_dusty_file_dir = config.dusty.dusty_file_dir
    default_cache_dir = config.dusty.npz_cache_dir

    parser = argparse.ArgumentParser(
        description="Generate DUSTY template grid models for SEDs"
    )

    # Template controls
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to Nugent spectral series file (e.g. sn2p_flux.v1.2.dat).")
    parser.add_argument("--template_tag", type=str, default="nugent_iip",
                        help="Tag to record in output rows.")
    
    # SED input
    parser.add_argument("--seds_pkl", type=str, default=None,
                        help="Pickle containing a single SED dict OR a list of SED dicts.")
    parser.add_argument("--sed_pickle_dir", type=str, default=None,
                        help="Directory containing *_tail_sed.pkl files (one SED dict per file).")
    parser.add_argument("--glob", type=str, default="*_tail_sed.pkl",
                        help="Glob used inside --sed_pickle_dir. Default '*_tail_sed.pkl'.")

    # Output
    parser.add_argument("workdir", type=str,
                        help="Base directory for outputs (models saved under workdir/template_grids/{oid}/).")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Where to write the combined CSV. Default: workdir/template_grids/grid_summary.csv")
    
    # DUSTY controls
    parser.add_argument("--tau_wav_micron", type=float, default=0.55,
                        help="Wavelength (micron) at which tau is specified.")
    parser.add_argument("--dtype", choices=["graphite", "silicate", "amorphous_carbon", "silicate_carbide"],
                        default="silicate",
                        help="Dust type to use.")
    parser.add_argument("--dusty_file_dir", type=str, default=default_dusty_file_dir,
                        help="Directory with DUSTY code files.")
    parser.add_argument("--thick_list", type=str, default="2.0",
                        help="Comma-separated shell thickness values, e.g. '2.0'.")
    
    # Grid values
    parser.add_argument("--tdust_list", type=str, default=None,
                        help="Comma-separated Tdust values in K.")
    parser.add_argument("--tau_list", type=str, default=None,
                        help="Comma-separated tau values.")
    parser.add_argument("--tstar_dummy", type=float, default=6000.0,
                        help="Dummy T* passed through API (template mode ignores it physically).")
    
    # Multiprocessing
    parser.add_argument("--n_workers", type=int, default=4)

    # Caching / runner behavior
    parser.add_argument("--cache_dir", type=str, default=default_cache_dir,
                        help="Directory for .npz cache.")
    parser.add_argument("--cache_ndigits", type=int, default=4)
    parser.add_argument("--cache_max", type=int, default=5000)
    parser.add_argument("--use_tmp", action="store_true")
    parser.add_argument("--force_rerun", action="store_true",
                        help="Ignore cache and rerun all models.")
    
    # Logging
    parser.add_argument("--loglevel", type=str, default="INFO",
                        help="Logging level.")
    parser.add_argument("--logfile", type=str, default=None,
                        help="Log file path.")

    args = parser.parse_args()

    logger = getLogger(args.loglevel, args.logfile)

    # -----------------
    # Resolve workdir 
    # -----------------
    workdir_path = Path(args.workdir)
    if not workdir_path.is_absolute():
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent.parent
        workdir_path = (project_root / args.workdir).resolve()

    # Create template_grids directory
    template_grids_dir = (workdir_path / "template_grids").resolve()
    template_grids_dir.mkdir(parents=True, exist_ok=True)

    # Output CSV
    if args.out_csv is None:
        out_csv = template_grids_dir / f"grid_summary_{args.template_tag}.csv"
    else:
        out_csv = Path(args.out_csv).resolve()

    # Cache dir
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else Path(config.dusty.npz_cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check dusty binary exists
    dusty_dir_abs = Path(args.dusty_file_dir).resolve()
    dusty_bin = dusty_dir_abs / "dusty"
    if not dusty_bin.exists():
        raise FileNotFoundError(f"Missing DUSTY binary at {dusty_bin}. Build it or fix --dusty_file_dir.")

    # -----------------------------
    # Load SEDs
    # -----------------------------
    sed_inputs = []
    if args.sed_pickle_dir is not None:
        sed_dir = Path(args.sed_pickle_dir).resolve()
        if not sed_dir.exists():
            raise FileNotFoundError(f"--sed_pickle_dir not found: {sed_dir}")

        pkl_paths = sorted(sed_dir.glob(args.glob))
        if not pkl_paths:
            raise RuntimeError(f"No files matching {args.glob} in {sed_dir}")

        for p in pkl_paths:
            with open(p, "rb") as f:
                obj = pickle.load(f)

            if not isinstance(obj, dict):
                raise RuntimeError(f"{p.name} did not contain a single SED dict.")

            if "sed" in obj and isinstance(obj["sed"], dict):
                sed = obj["sed"]
                oid = obj.get("oid", sed.get("oid", "unknown"))
                phase = sed.get("phase_days", obj.get("phase", None))
            else:
                oid = obj.get("oid", "unknown")
                phase = obj.get("phase_days", obj.get("phase", None))

            if phase is None or not np.isfinite(float(phase)):
                logger.warning(f"Skipping {p.name} - invalid phase")
                continue

            sed_inputs.append((oid, float(phase)))

    else:
        if args.seds_pkl is None:
            raise RuntimeError("Provide either --sed_pickle_dir or --seds_pkl.")

        seds_p = Path(args.seds_pkl).resolve()
        with open(seds_p, "rb") as f:
            obj = pickle.load(f)

        if isinstance(obj, dict):
            oid = obj.get("oid", "unknown")
            sed = obj.get("sed", obj)
            phase = sed.get("phase_days", obj.get("phase", None))
            if phase and np.isfinite(float(phase)):
                sed_inputs.append((oid, float(phase)))
        elif isinstance(obj, (list, tuple)):
            for i, sed_dict in enumerate(obj):
                oid = sed_dict.get("oid", f"unknown_{i}")
                phase = sed_dict.get("phase_days", sed_dict.get("phase", None))
                if phase and np.isfinite(float(phase)):
                    sed_inputs.append((oid, float(phase)))

    if not sed_inputs:
        raise RuntimeError("No SEDs loaded.")

    logger.info(f"Loaded {len(sed_inputs)} SED(s)")

    # --------------
    # Build grids 
    # --------------
    if args.tdust_list:
        tdust_values = _parse_list(args.tdust_list)
    else:
        tdust_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    if args.tau_list:
        tau_values = _parse_list(args.tau_list)
    else:
        tau_values = np.r_[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 1.5, 2.0, 3.0, 4.0]
    if args.thick_list:
        thick_values = _parse_list(args.thick_list)
    else:
        thick_list=[2.0]

    # ----------------
    # Runner kwargs
    # ----------------
    runner_kwargs = dict(
        base_workdir=str(template_grids_dir),
        dusty_file_dir=str(dusty_dir_abs),
        dust_type=args.dtype,
        shell_thickness=float(thick_values[0]) if len(thick_values) == 1 else 2.0,  
        tau_wavelength_microns=float(args.tau_wav_micron),
        cache_dir=str(cache_dir),
        cache_ndigits=int(args.cache_ndigits),
        cache_max=int(args.cache_max),
        use_tmp=bool(args.use_tmp),
        run_tag=f"tmplgrid_{args.template_tag}",
        blackbody=True, 
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
        runner_kwargs["run_tag"],
    )

    # -------------
    # Check cache 
    # -------------
    completed = set()
    
    if not args.force_rerun and out_csv.exists():
        prev = pd.read_csv(out_csv)
        prev_ok = prev[(prev["ierror"] == 0) & prev["npz_path"].notna()]
        for _, r in prev_ok.iterrows():
            _, key = make_leaf_and_key(
                r["tstar_dummy"], r["tdust"], r["tau"], r["shell_thickness"],
                r["phase_days"], r["template_tag"], args.cache_ndigits
            )
            completed.add(key)
        logger.info(f"Found {len(completed)} completed models to skip.")

    # ----------------
    # Build jobs list 
    # ----------------
    jobs = []
    skipped = 0
    
    for oid, phase in sed_inputs:
        for thick, tdust, tau in it.product(thick_values, tdust_values, tau_values):
            _, key = make_leaf_and_key(
                args.tstar_dummy, tdust, tau, thick, phase, args.template_tag, args.cache_ndigits
            )
            
            if key in completed:
                skipped += 1
                continue
            
            jobs.append((
                float(args.tstar_dummy), float(tdust), float(tau), float(thick), float(phase), oid,
                str(Path(args.template_path).resolve()), args.template_tag,
                runner_kwargs, runner_sig, args.cache_ndigits
            ))

    logger.info(f"Skipping {skipped} cached models, running {len(jobs)} new ones.")

    if len(jobs) == 0:
        logger.info("No jobs to run (all cached). Exiting.")
        return

    # --------------------
    # Parallel execution 
    # --------------------
    max_workers = min(os.cpu_count() or 2, int(args.n_workers))
    
    results_summary = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_model, job) for job in jobs]
        for fut in as_completed(futures):
            res = fut.result()
            results_summary.append(res)
            status = "OK" if res.get("ierror", 1) == 0 else "ERROR"
            logger.info(
                f"{status}  OID={res['oid']} phase={res['phase_days']:.2f} "
                f"Td={res['tdust']} tau={res['tau']} thick={res['shell_thickness']} "
                f"-> npz={res.get('npz_path', 'N/A')}"
            )

    # ------------------
    # Write summary CSV 
    # ------------------
    if results_summary:
        # Load existing results if not force_rerun
        if not args.force_rerun and out_csv.exists():
            existing = pd.read_csv(out_csv)
            new_df = pd.DataFrame(results_summary)
            df = pd.concat([existing, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(results_summary)
        
        df = df.sort_values(["oid", "phase_days", "tdust", "tau", "shell_thickness"])
        df.to_csv(out_csv, index=False)
        
        logger.info(f"Wrote model grid summary to {out_csv}")
    else:
        logger.warning("No results to write.")

if __name__ == "__main__":
    main()