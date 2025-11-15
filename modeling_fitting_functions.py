import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re
from astropy.io import ascii
from astropy import constants as const
import matplotlib.pyplot as plt
import pandas as pd
from SED_functions import plot_sed, _prepare_sed_xy
import matplotlib.colors as mcolors

# ------------------------------
# Containers for DUSTY model
# ------------------------------

class DustyModel:
    def __init__(self, folder, Tstar, Tdust, tau, lam_um, lamFlam):
        
        self.folder = Path(folder)
        self.Tstar = float(Tstar)
        self.Tdust = float(Tdust)
        self.tau = float(tau)
        self.lam_um = np.array(lam_um, float)
        self.lamFlam = np.array(lamFlam, float)

        # filled after fitting
        self.scale: Optional[float] = None
        self.chi2: Optional[float] = None
        self.dof: Optional[int] = None
        self.chi2_red: Optional[float] = None
        self.x_plot: Optional[np.ndarray] = None
        self.y_scaled: Optional[np.ndarray] = None

def load_dusty_grid(grid_dir):
    """
    Looks for subfolders named like 'Tstar_4000_Tdust_1100_tau_0_03' containing 'sed.dat'
    with two columns: lam[μm], flux[=λFλ].
    """

    grid_dir = Path(grid_dir)
    rx = re.compile(r"Tstar_(\d+)_Tdust_(\d+)_tau_([0-9_]+)$")

    models: List[DustyModel] = []
    for sub in sorted(grid_dir.iterdir()):
        if not sub.is_dir():
            continue
        match = rx.fullmatch(sub.name)
        if not match:
            continue
        sed_path = sub / "sed.dat"
        if not sed_path.exists():
            continue

        tstar = float(match.group(1))
        tdust = float(match.group(2))
        tau = float(match.group(3).replace("_", "."))

        table = ascii.read(sed_path, names=["lam", "flux"], comment="#", fast_reader=False)
        lam_um = np.array(table["lam"], float)
        lamFlam = np.array(table["flux"], float)
        models.append(DustyModel(sub, tstar, tdust, tau, lam_um, lamFlam))

    if not models:
        raise RuntimeError(f"No DUSTY models with sed.dat found under {grid_dir}")
    
    return models

# ------------------------------
# Fit model scale to SED
# ------------------------------

def fit_scale_to_sed(model, sed, y_mode="Flam", use_weights=True):
    """
    Find best-fit scale factor for DUSTY model to SED using least-squares.

    Parameters
    ----------
    dusty_data : dict or astropy.Table
        Must have 'lam' [µm] and 'flux' [arbitrary or erg/s/cm^2/µm].
    sed : dict
        SED dictionary containing 'lam', 'nu', 'Fnu', 'eFnu', etc.
    y_mode : str
        'Flam' or 'Fnu' — determines whether we compare in λFλ or Fν space.

    Returns
    -------
    scale : float
        Multiplicative scale factor for DUSTY flux.
    scaled_dusty : dict
        Dictionary with 'lam' and 'flux' arrays scaled appropriately.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)

    if y_mode == "Flam":
        x_mod = model.lam_um
        y_mod = model.lamFlam

        # find overlap
        mask = (x_mod >= x_sed.min()) & (x_mod <= x_sed.max())

        if not np.any(mask):
            model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
            model.x_plot, model.y_scaled = x_mod, y_mod
            return model
        
        y_data = np.interp(x_mod[mask], x_sed, y_sed)
        if use_weights and np.any(ey_sed > 0):
            ey_data = np.interp(x_mod[mask], x_sed, ey_sed)
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
            a = np.sum(weights * y_data * y_mod[mask]) / np.sum(weights * y_mod[mask]**2)
            chi2 = np.sum(weights * (y_data - a*y_mod[mask])**2)
        else:
            a = np.sum(y_data * y_mod[mask]) / np.sum(y_mod[mask]**2)
            chi2 = np.sum((y_data - a*y_mod[mask])**2)

        dof = max(np.count_nonzero(mask)-1, 0)

        model.scale = float(a)
        model.chi2 = float(chi2)
        model.dof = int(dof)
        model.chi2_red = float(chi2/dof) if dof > 0 else np.inf
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

        mask = (nu_mod >= x_sed.min()) & (nu_mod <= x_sed.max())
        
        if not np.any(mask):
            model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
            model.x_plot, model.y_scaled = nu_mod, Fnu_mod
            return model

        y_data = np.interp(nu_mod[mask], x_sed, y_sed)
        if use_weights and np.any(ey_sed > 0):
            ey_data = np.interp(nu_mod[mask], x_sed, ey_sed)
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
            a = np.sum(weights * y_data * Fnu_mod[mask]) / np.sum(weights * Fnu_mod[mask]**2)
            chi2 = np.sum(weights * (y_data - a*Fnu_mod[mask])**2)
        else:
            a = np.sum(y_data * Fnu_mod[mask]) / np.sum(Fnu_mod[mask]**2)
            chi2 = np.sum((y_data - a*Fnu_mod[mask])**2)

        dof = max(np.count_nonzero(mask)-1, 0)

        model.scale = float(a)
        model.chi2 = float(chi2)
        model.dof = int(dof)
        model.chi2_red = float(chi2/dof) if dof > 0 else np.inf
        model.x_plot = nu_mod
        model.y_scaled = a * Fnu_mod

        return model
    
# ------------------------------
# Fit a full grid
# ------------------------------

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

    model_map = {}

    for m in models:
        fit_scale_to_sed(m, sed, y_mode=y_mode, use_weights=use_weights)
        rows.append(dict(
            folder=str(m.folder),
            Tstar=m.Tstar, Tdust=m.Tdust, tau=m.tau,
            scale=m.scale, chi2=m.chi2, dof=m.dof,
            chi2_red=m.chi2_red
        ))
        model_map[str(m.folder)] = m

    df = pd.DataFrame(rows).sort_values("chi2_red")
    df._models = model_map

    return df

def plot_best_fit_dusty_model(df, sed, y_mode="Flam", top_n=10, keep_sed_limits=False,
                              y_padding_frac=0.15, logx=True, logy=True, secax=False, 
                              savepath=None):
    """
    Plot the best N models (after scaling) over the SED.
    """

    # fun color stuff
    base = mcolors.to_rgb("darkmagenta")

    def _lighten(rgb, t):
        """t in [0,1]; 0 = original, 1 = white"""
        return tuple((1 - t)*c + t*1.0 for c in rgb)

    def _darken(rgb, t):
        """t in [0,1]; 0 = original, 1 = black"""
        return tuple((1 - t)*c for c in rgb)
    N = top_n
    palette = ([_darken(base, t)  for t in np.linspace(0.0, 0.75, N//2)] +
           [_lighten(base, t) for t in np.linspace(0.0, 0.75, N - N//2)])

    x_sed, y_sed, ey_sed, xlab, ylab = _prepare_sed_xy(sed, y_mode=y_mode)

    fig, ax = plt.subplots(figsize=(8,6))

    best = df.head(min(top_n, len(df)))
    models = df._models

    for i, (_, r) in enumerate(best.iterrows()):
        m = models[r['folder']]
        ax.plot(m.x_plot, m.y_scaled, lw=2, color=palette[i % len(palette)],
                label=f"T*: {m.Tstar} K, T_dust: {m.Tdust} K, tau: {m.tau}")

    # SED data
    plot_sed(sed, ax=ax, y_mode=y_mode, logx=logx, logy=logy, secax=secax, savepath=None)

    if keep_sed_limits:
        ax.set_ylim(1e-14, 1e-12)
        ax.set_xlim(0.2, 12)
    elif not keep_sed_limits:
        # pad y-limits a bit
        vals = [np.nanmin(y_sed), np.nanmax(y_sed)]
        for _, r in best.iterrows():
            m = models[r['folder']]
            vals += [np.nanmin(m.y_scaled), np.nanmax(m.y_scaled)]
        ymin, ymax = np.nanmin(vals), np.nanmax(vals)

        if np.isfinite(ymin) and np.isfinite(ymax) and ymin > 0:
            pad = (np.log10(ymax) - np.log10(ymin)) * y_padding_frac
            ax.set_ylim(10**(np.log10(ymin) - pad), 10**(np.log10(ymax) + pad))

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Saved plot to {savepath}")
    else:
        plt.show()