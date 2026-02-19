import numpy as np
import pandas as pd
from pathlib import Path
from astropy import constants as const

from ..sed.build import _prepare_sed_xy
from .model import DustyModel, load_dusty_grid


def compute_chi2(lam_um, lamFlam, sed, a=None, y_mode="Flam", use_weights=True):
    """
    Compute chi-squared for a DUSTY model with a provided scale factor ``a``.
    Use ``fit_scale_to_sed`` first if you need to determine the best-fit scale.
    """
    if a is None:
        raise ValueError("compute_chi2 requires a scale factor 'a'; call fit_scale_to_sed first.")

    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)
    
    if y_mode == "Flam":
        x_mod = lam_um
        y_mod = lamFlam * a
        
        # overlap in x
        mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
        x_data = x_sed[mask_data]
        y_data = y_sed[mask_data]
        ey_data = ey_sed[mask_data]
        
        if len(x_data) == 0:
            return a, np.inf
        
        # interpolate model onto observed wavelengths
        y_mod_on_data = np.interp(x_data, x_mod, y_mod)
        
    elif y_mode == "Fnu":
        lam_cm = lam_um * 1e-4
        nu_mod = const.c.cgs.value / lam_cm  # Hz
        Fnu_mod = (lamFlam * a) * (lam_cm / const.c.cgs.value)
        
        order = np.argsort(nu_mod)
        nu_mod = nu_mod[order]
        Fnu_mod = Fnu_mod[order]
        
        # overlap in x
        mask_data = (x_sed >= nu_mod.min()) & (x_sed <= nu_mod.max())
        x_data = x_sed[mask_data]
        y_data = y_sed[mask_data]
        ey_data = ey_sed[mask_data]
        
        if len(x_data) == 0:
            return a, np.inf
        
        # interpolate model onto observed wavelengths
        y_mod_on_data = np.interp(x_data, nu_mod, Fnu_mod)
    
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")
    
    if use_weights and np.any(ey_data > 0):
        weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
        chi2 = np.sum(weights * (y_data - y_mod_on_data)**2)
    else:
        chi2 = np.sum((y_data - y_mod_on_data)**2)
    
    return a, float(chi2)


def fit_scale_to_sed(model, sed, y_mode="Flam", use_weights=True):
    """
    Find best-fit scale factor for DUSTY model to match SED using least-squares.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)

    if y_mode == "Flam":
        x_mod = model.lam_um
        y_mod = model.lamFlam

        # overlap in x
        mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
        x_data = x_sed[mask_data]
        y_data = y_sed[mask_data]
        ey_data = ey_sed[mask_data]

        if len(x_data) == 0:
            model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
            model.x_plot, model.y_scaled = x_mod, y_mod
            return model
        
        # interpolate model onto observed wavelengths
        y_mod_on_data = np.interp(x_data, x_mod, y_mod)

        # LS fit for a
        if use_weights and np.any(ey_data > 0):
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
            a = np.sum(weights * y_data * y_mod_on_data) / np.sum(weights * y_mod_on_data**2)
            chi2 = np.sum(weights * (y_data - a*y_mod_on_data)**2)
        else:
            a = np.sum(y_data * y_mod_on_data) / np.sum(y_mod_on_data**2)
            chi2 = np.sum((y_data - a*y_mod_on_data)**2)

        N = len(y_data)
        dof = max(N - 1, 0)

        model.scale = float(a)
        model.chi2 = float(chi2)
        model.dof = int(dof)
        model.chi2_red = float(chi2 / dof) if dof > 0 else np.inf
        model.x_plot = x_mod
        model.y_scaled = a * y_mod
        
        return model
    
    elif y_mode == "Fnu":
        lam_cm = model.lam_um * 1e-4
        nu_mod = const.c.cgs.value / lam_cm  # Hz
        Fnu_mod = model.lamFlam * (lam_cm / const.c.cgs.value)

        order = np.argsort(nu_mod)
        nu_mod = nu_mod[order]
        Fnu_mod = Fnu_mod[order]

        # overlap in x
        mask_data = (x_sed >= nu_mod.min()) & (x_sed <= nu_mod.max())
        x_data = x_sed[mask_data]
        y_data = y_sed[mask_data]
        ey_data = ey_sed[mask_data]

        if len(x_data) == 0:
            model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
            model.x_plot, model.y_scaled = nu_mod, Fnu_mod
            return model

        # interpolate model onto observed wavelengths
        y_mod_on_data = np.interp(x_data, nu_mod, Fnu_mod)

        # LS fit for a
        if use_weights and np.any(ey_data > 0):
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
            a = np.sum(weights * y_data * y_mod_on_data) / np.sum(weights * y_mod_on_data**2)
            chi2 = np.sum(weights * (y_data - a*y_mod_on_data)**2)
        else:
            a = np.sum(y_data * y_mod_on_data) / np.sum(y_mod_on_data**2)
            chi2 = np.sum((y_data - a*y_mod_on_data)**2)

        N = len(y_data)
        dof = max(N - 1, 0)

        model.scale = float(a)
        model.chi2 = float(chi2)
        model.dof = int(dof)
        model.chi2_red = float(chi2 / dof) if dof > 0 else np.inf
        model.x_plot = nu_mod
        model.y_scaled = a * Fnu_mod

        return model
    
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")

def fit_grid_to_sed(grid_csv, sed, shell_thickness=None, y_mode="Flam", use_weights=True,
                    template_tag=None, top_k=None, max_tstar=6000.0):
    """
    Fitting function for both blackbody and template grids.

    Parameters
    ----------
    grid_csv : str or Path
        CSV file containing grid model information.
    sed : dict
        SED dictionary containing 'lam', 'nu', 'Fnu', 'eFnu', etc.
    shell_thickness : float or None
        Shell thickness to filter by. If None, uses all models in CSV.
    y_mode : str
        'Flam' or 'Fnu' 
    use_weights : bool
        Whether to use SED uncertainties as weights in fitting.
    template_tag : str or None
        If fitting a template grid, filter by this template_tag value. 
    top_k : int or None
        Only load and fit top K models from CSV.
    
    Returns
    -------
    DataFrame
        DataFrame with fit results sorted by chi2_red, including a '_models' 
        attribute mapping folders to DustyModel instances.
    """

    grid_csv = Path(grid_csv)
    if not grid_csv.exists():
        raise FileNotFoundError(f"Grid CSV file not found: {grid_csv}")
    
    is_template = "template" in grid_csv.name or 'nugent' in grid_csv.name

    # load models from csv
    if is_template:
        oid = sed.get('oid', None)
        if oid is None:
            raise ValueError("SED dictionary must contain 'oid' key for template grid fitting")
        
        if template_tag is None and 'nugent' in grid_csv.name:
            template_tag = "nugent_iip"

        models = load_dusty_grid(
            csv_path=grid_csv,
            prefer_npz=True,
            oid=oid,
            template_tag=template_tag,
            top_k=top_k
            )
    else:
        models = load_dusty_grid(
            csv_path=grid_csv,
            prefer_npz=True
            )

    if shell_thickness is not None:
        models = [m for m in models if np.isclose(m.shell_thickness, shell_thickness, atol=0.01)]
    
    if not is_template and max_tstar is not None:
        n_before = len(models)
        models = [m for m in models if m.Tstar <= max_tstar]
        n_after = len(models)
        if n_before > n_after:
            print(f"Filtered out {n_before - n_after} models with Tstar > {max_tstar} K")

    if not models:
        raise RuntimeError(
            f"No models found in {grid_csv}" + 
            (f" for shell_thickness={shell_thickness}" if shell_thickness else "") +
            (f" and oid={sed.get('oid')}" if is_template else "")
        )

    # fit all models to SED
    rows = []
    model_map = {}
    oid_for_output = sed.get('oid', 'unknown')

    for m in models:
        fit_scale_to_sed(m, sed, y_mode=y_mode, use_weights=use_weights)
        
        row = dict(
            folder=str(m.folder), 
            oid=oid_for_output,
            tdust=m.Tdust,
            tau=m.tau,
            shell_thickness=m.shell_thickness,
            scale=m.scale,
            chi2=m.chi2,
            dof=m.dof,
            chi2_red=m.chi2_red
        )

        if m.npz_path is not None:
            row['npz_path'] = m.npz_path
        if m.outpath is not None:
            row['outpath'] = m.outpath
        
        if not is_template:
            row['tstar'] = m.Tstar
        if is_template:
            row['tstar_dummy'] = m.Tstar  # placeholder for template grids
            row['phase_days'] = m.phase_days
            row['template_tag'] = m.template_tag
        
        rows.append(row)
        model_map[str(m.folder)] = m
    
    # create DataFrame and sort by chi2_red
    df = pd.DataFrame(rows).sort_values("chi2_red").reset_index(drop=True)
    df._models = model_map

    return df

# def fit_blackbody_grid_to_sed(grid_csv, sed, y_mode="Flam", use_weights=True):
#     """
#     Load DUSTY model grid from `grid_dir` and fit each model to the SED.

#     Parameters
#     ----------
#     grid_csv : str or Path
#         CSV file containing grid model information.
#     sed : dict
#         SED dictionary containing 'lam', 'nu', 'Fnu', 'eFnu', etc.
#     y_mode : str
#         'Flam' or 'Fnu' — determines whether we compare in λFλ or Fν space.
#     use_weights : bool
#         Whether to use SED uncertainties as weights in fitting.

#     Returns
#     -------
#     models : list of DustyModel
#         List of fitted DustyModel instances with scale and chi2 attributes filled.
#     """

#     df_all = pd.read_csv(grid_csv)

#     models = load_dusty_grid(grid_dir)
#     rows = []

#     oid = sed['oid']

#     model_map = {}

#     for m in models:
#         fit_scale_to_sed(m, sed, y_mode=y_mode, use_weights=use_weights)
#         rows.append(dict(
#             folder=str(m.folder), oid=oid,
#             tstar=m.Tstar, tdust=m.Tdust, tau=m.tau,
#             scale=m.scale, chi2=m.chi2, dof=m.dof,
#             chi2_red=m.chi2_red
#         ))
#         model_map[str(m.folder)] = m

#     df = pd.DataFrame(rows).sort_values("chi2_red")
#     df._models = model_map

#     return df

# def fit_template_grid_to_sed(template_grid_csv, sed, runner, template,
#                              template_tag="nugent_iip", y_mode="Flam", use_weights=True,
#                              top_k_build=None, folder_prefix="TEMPLATE"):
#     """
#     Convert a template-based grid CSV into DUSTY models, fit to SED.
#     """
#     df_all = pd.read_csv(template_grid_csv)
#     oid = sed.get('oid', None)
#     if oid is None:
#         raise ValueError("SED dictionary must contain 'oid' key")
#     phase = sed.get("phase_days", sed.get("phase", None))
#     if phase is None or not np.isfinite(float(phase)):
#         raise ValueError("sed must include finite 'phase_days'")
    
#     # select SN by oid
#     df = df_all[df_all["oid"].astype(str) == str(oid)].copy()
#     if df.empty:
#         raise RuntimeError(f"No template-grid rows found for oid={oid}")
    
#     if "template_tag" in df.columns and template_tag is not None:
#         df = df[df["template_tag"].astype(str) == str(template_tag)].copy()
#         if df.empty:
#             raise RuntimeError(f"No rows for oid={oid} with template_tag={template_tag}")
    
#     # could add logic later in case multiple SEDs per SN
#     sort_cols = [c for c in ["chi2_red", "chi2"] if c in df.columns]
#     if sort_cols:
#         df = df.sort_values(sort_cols).reset_index(drop=True)
    
#     if top_k_build is not None:
#         df_build = df.head(int(top_k_build)).copy()
#     else:
#         df_build = df.copy()
    
#     model_map = {}
    
#     # Initialize new columns in df_build
#     df_build['folder'] = None
#     df_build['scale'] = np.nan
#     df_build['chi2'] = np.nan
#     df_build['dof'] = np.nan
#     df_build['chi2_red'] = np.nan
    
#     for idx, r in df_build.iterrows():
#         tdust = float(r["tdust"])
#         tau = float(r["tau"])
#         thick = float(r["shell_thickness"]) if "shell_thickness" in r else float(r.get("thick", 2.0))
#         tstar = float(r["tstar_dummy"]) if "tstar_dummy" in r else float(r.get("tstar", 6000.0))
        
#         lam_um, lamFlam, r1 = runner.evaluate_model(
#             tstar=tstar,
#             tdust=tdust,
#             tau=tau,
#             shell_thickness=thick,
#             template=template,
#             phase_days=float(phase),
#             template_tag=str(template_tag),
#         )
        
#         folder = f"{folder_prefix}_{oid}_row{idx}"
        
#         m = DustyModel(
#             folder=folder,
#             Tstar=tstar,
#             Tdust=tdust,
#             tau=tau,
#             shell_thickness=thick,
#             lam_um=lam_um,
#             lamFlam=lamFlam,
#         )
        
#         fit_scale_to_sed(m, sed, y_mode=y_mode, use_weights=use_weights)
        
#         model_map[folder] = m
        
#         # Update the dataframe with fit results
#         df_build.loc[idx, 'folder'] = folder
#         df_build.loc[idx, 'scale'] = m.scale
#         df_build.loc[idx, 'chi2'] = m.chi2
#         df_build.loc[idx, 'dof'] = m.dof
#         df_build.loc[idx, 'chi2_red'] = m.chi2_red
    
#     if df_build.empty:
#         raise RuntimeError(f"Could not build any DustyModel objects for oid={oid}")
    
#     # Sort by chi2_red
#     df_out = df_build.sort_values("chi2_red").reset_index(drop=True)
#     df_out._models = model_map
    
#     return df_out