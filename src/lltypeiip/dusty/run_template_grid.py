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
import re
import pickle
import pandas as pd
from astropy import constants as const

from ..config import config
from ..sed.build import _prepare_sed_xy
from ..dusty.runner import DustyRunner
from ..dusty.custom_spectrum import NugentIIPSeries

from pydusty.dusty import DustyCustomInputSpectrum
from pydusty.parameters import Parameter, DustyParameters
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
    return [float(x) for x in s.split(",") if x.strip()]

def _ls_scale_and_chi2(lam_um, lamFlam, sed, y_mode="Flam", use_weights=True):
    """
    Analytic best-fit scale and chi2 on the SED grid.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)

    if y_mode == "Flam":
        x_mod = np.asarray(lam_um, float)
        y_mod = np.asarray(lamFlam, float)
    elif y_mode == "Fnu":
        lam_cm = np.asarray(lam_um, float) * 1e-4
        nu_mod = const.c.cgs.value / lam_cm
        Fnu_mod = np.asarray(lamFlam, float) * (lam_cm / const.c.cgs.value)
        order = np.argsort(nu_mod)
        x_mod = nu_mod[order]
        y_mod = Fnu_mod[order]
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")

    mask = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
    if not np.any(mask):
        return np.nan, np.inf

    x_data = x_sed[mask]
    y_data = y_sed[mask]
    ey_data = ey_sed[mask]

    y_mod_on_data = np.interp(x_data, x_mod, y_mod)

    if use_weights and np.any(np.isfinite(ey_data)) and np.any(ey_data > 0):
        w = 1.0 / np.clip(ey_data, 1e-99, np.inf) ** 2
        a = np.sum(w * y_data * y_mod_on_data) / np.sum(w * y_mod_on_data ** 2)
        chi2 = np.sum(w * (y_data - a * y_mod_on_data) ** 2)
    else:
        a = np.sum(y_data * y_mod_on_data) / np.sum(y_mod_on_data ** 2)
        chi2 = np.sum((y_data - a * y_mod_on_data) ** 2)

    return float(a), float(chi2)

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

def _get_runner(args_sig: tuple, runner_kwargs: dict):
    """
    Create DustyRunner once per worker process.
    """
    global _G
    if (_G["runner"] is None) or (_G["runner_sig"] != args_sig):
        _G["runner"] = DustyRunner(**runner_kwargs)
        _G["runner_sig"] = args_sig
    return _G["runner"]


def run_single_sed(job):
    """
    Worker: evaluate grid for one SED epoch and return list of result rows.
    Mirrors run_grid.py idea of "unit of work", but per-SED instead of per-model folder.
    """
    (sed,
     tdust_values, tau_values, thick_values,
     tstar_dummy, template_path, template_tag,
     y_mode, use_weights,
     runner_kwargs, runner_sig) = job

    phase = sed.get("phase_days", None)
    mjd = sed.get("mjd", np.nan)
    oid = sed.get("oid", "unknown")

    if phase is None or (not np.isfinite(float(phase))):
        return []
    
    template = _get_template(template_path)
    runner = _get_runner(runner_sig, runner_kwargs)

    rows = []
    dof = max(len(sed.get("bands", [])) - 1, 1)

    for thick, tdust, tau in it.product(thick_values, tdust_values, tau_values):
        lam_um, lamFlam, r1 = runner.evaluate_model(
            tstar=float(tstar_dummy),
            tdust=float(tdust),
            tau=float(tau),
            shell_thickness=float(thick),
            template=template,
            phase_days=float(phase),
            template_tag=str(template_tag),
        )

        if lam_um is None or lamFlam is None:
            continue

        a, chi2 = _ls_scale_and_chi2(lam_um, lamFlam, sed, y_mode=y_mode, use_weights=use_weights)

        if not np.isfinite(chi2):
            continue

        rows.append(dict(
            oid=str(oid),
            mjd=float(mjd),
            phase_days=float(phase),
            template_tag=str(template_tag),
            tstar_dummy=float(tstar_dummy),
            tdust=float(tdust),
            tau=float(tau),
            shell_thickness=float(thick),
            scale=float(a),
            chi2=float(chi2),
            chi2_red=float(chi2 / dof),
            r1=float(r1),
            ierror=0,
            error=None,
        ))

    return rows

def main():
    default_dusty_file_dir = config.dusty.dusty_file_dir

    parser = argparse.ArgumentParser()

    # template controls
    parser.add_argument("--template_path", type=str, required=True,
                        help="Path to Nugent spectral series file (e.g. sn2p_flux.v1.2.dat).")
    parser.add_argument("--template_tag", type=str, default="nugent_iip",
                        help="Tag to record in output rows.")
    
    # SED input
    parser.add_argument("--seds_pkl", type=str, required=True,
                        help="Pickle file containing list of SED dicts (each must have phase_days).")
    
    # output
    parser.add_argument("workdir", type=str,
                        help="Directory used for DUSTY work + caches + outputs.")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Where to write the combined CSV. Default under workdir.")
    parser.add_argument("--write_per_epoch_csv", action="store_true",
                        help="Also write one CSV per SED epoch under workdir.")
    
    # dusty controls (mirror run_grid.py)
    parser.add_argument("--tau_wav_micron", type=float, default=0.55,
                        help="Wavelength (micron) at which tau is specified.")
    parser.add_argument("--dtype", choices=["graphite", "silicate", "amorphous_carbon", "silicate_carbide"],
                        default="silicate",
                        help="Dust type to use.")
    parser.add_argument("--dusty_file_dir", type=str, default=default_dusty_file_dir,
                        help="Directory with DUSTY code files.")
    parser.add_argument("--thick_list", type=str, default="2.0",
                        help="Comma-separated shell thickness values, e.g. '2.0'.")
    
    # grid values
    parser.add_argument("--tdust_list", type=str, default=None,
                        help="Comma-separated Tdust values in K.")
    parser.add_argument("--tau_list", type=str, default=None,
                        help="Comma-separated tau values.")
    parser.add_argument("--tstar_dummy", type=float, default=6000.0,
                        help="Dummy T* passed through API (template mode ignores it physically).")
    
    # scoring
    parser.add_argument("--y_mode", choices=["Flam", "Fnu"], default="Flam")
    parser.add_argument("--use_weights", action="store_true")

    
    # multiprocessing
    parser.add_argument("--n_workers", type=int, default=4)

    # caching / runner behavior
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Directory for .npz cache. Default under workdir.")
    parser.add_argument("--cache_ndigits", type=int, default=4)
    parser.add_argument("--cache_max", type=int, default=5000)
    parser.add_argument("--use_tmp", action="store_true")

    args = parser.parse_args()

    # -----------------
    # Resolve workdir 
    # -----------------
    workdir_path = Path(args.workdir)
    if not workdir_path.is_absolute():
        script_dir = Path(__file__).parent.resolve()
        project_root = script_dir.parent.parent.parent
        workdir_path = (project_root / args.workdir).resolve()

    workdir = workdir_path.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    # output CSV
    if args.out_csv is None:
        out_csv = workdir / f"template_grid_summary_{args.template_tag}.csv"
    else:
        out_csv = Path(args.out_csv).resolve()

    # cache dir
    cache_dir = Path(args.cache_dir).resolve() if args.cache_dir else (workdir / "dusty_npz_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # check dusty binary exists
    dusty_dir_abs = Path(args.dusty_file_dir).resolve()
    dusty_bin = dusty_dir_abs / "dusty"
    if not dusty_bin.exists():
        raise FileNotFoundError(f"Missing DUSTY binary at {dusty_bin}. Build it or fix --dusty_file_dir.")

    
    # -----------------------------
    # Load SEDs
    # -----------------------------
    seds_p = Path(args.seds_pkl).resolve()
    with open(seds_p, "rb") as f:
        seds = pickle.load(f)

    if not isinstance(seds, (list, tuple)) or len(seds) == 0:
        raise RuntimeError("seds_pkl did not contain a non-empty list of SED dicts.")

    # --------------
    # Build grids 
    # --------------
    if args.tdust_list:
        tdust_values = _parse_list(args.tdust_list)
    else:
        tdust_values = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

    if args.tau_list:
        tau_values = _parse_list(args.tau_list)
    else:
        tau_values = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0]

    thick_values = _parse_list(args.thick_list)

    # -----------------------------
    # Runner kwargs (passed into workers)
    # -----------------------------
    runner_kwargs = dict(
        base_workdir=str(workdir),
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

    # --------------------
    # Parallel execution
    # --------------------
    jobs = []
    for sed in seds:
        jobs.append(
            (sed,
             tdust_values, tau_values, thick_values,
             float(args.tstar_dummy), str(Path(args.template_path).resolve()), args.template_tag,
             args.y_mode, bool(args.use_weights),
             runner_kwargs, runner_sig)
        )

    max_workers = min(os.cpu_count() or 2, int(args.n_workers))
    all_rows = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(run_single_sed, job) for job in jobs]
        for fut in as_completed(futures):
            rows = fut.result()
            if rows:
                all_rows.extend(rows)

                if args.write_per_epoch_csv:
                    mjd0 = rows[0]["mjd"]
                    oid = rows[0]["oid"]
                    phase = rows[0]["phase_days"]
                    df_epoch = pd.DataFrame(rows).sort_values("chi2_red")
                    leaf = f"{oid}_mjd_{mjd0:.3f}_phase_{phase:.2f}".replace(".", "_")
                    df_epoch.to_csv(workdir / f"grid_{leaf}.csv", index=False)

    if not all_rows:
        raise RuntimeError("No grid rows produced. Check phase_days and template coverage.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["oid", "mjd", "chi2_red"])

    df.to_csv(out_csv, index=False)

    best = df.groupby(["oid", "mjd"], as_index=False).first()
    best_csv = out_csv.with_name(out_csv.stem + "_best_per_epoch.csv")
    best.to_csv(best_csv, index=False)

    print(f"[template-grid] wrote: {out_csv}")
    print(f"[template-grid] wrote: {best_csv}")

if __name__ == "__main__":
    main()