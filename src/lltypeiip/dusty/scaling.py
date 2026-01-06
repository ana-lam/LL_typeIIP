import numpy as np
import pandas as pd
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

def fit_grid_to_sed(grid_dir, sed, y_mode="Flam", use_weights=True):
    """
    Load DUSTY model grid from `grid_dir` and fit each model to the SED.

    Parameters
    ----------
    grid_dir : str or Path
        Directory containing subfolders with DUSTY models.
    sed : dict
        SED dictionary containing 'lam', 'nu', 'Fnu', 'eFnu', etc.
    y_mode : str
        'Flam' or 'Fnu' — determines whether we compare in λFλ or Fν space.
    use_weights : bool
        Whether to use SED uncertainties as weights in fitting.

    Returns
    -------
    models : list of DustyModel
        List of fitted DustyModel instances with scale and chi2 attributes filled.
    """
    models = load_dusty_grid(grid_dir)
    rows = []

    oid = sed['oid']

    model_map = {}

    for m in models:
        fit_scale_to_sed(m, sed, y_mode=y_mode, use_weights=use_weights)
        rows.append(dict(
            folder=str(m.folder), oid=oid,
            tstar=m.Tstar, tdust=m.Tdust, tau=m.tau,
            scale=m.scale, chi2=m.chi2, dof=m.dof,
            chi2_red=m.chi2_red
        ))
        model_map[str(m.folder)] = m

    df = pd.DataFrame(rows).sort_values("chi2_red")
    df._models = model_map

    return df