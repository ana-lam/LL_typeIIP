import numpy as np
from pathlib import Path
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
    def __init__(self, folder, Tstar, Tdust, tau, shell_thickness, lam_um, lamFlam):
        
        self.folder = Path(folder)
        self.Tstar = float(Tstar)
        self.Tdust = float(Tdust)
        self.tau = float(tau)
        self.shell_thickness = float(shell_thickness)
        self.lam_um = np.array(lam_um, float)
        self.lamFlam = np.array(lamFlam, float)

        # filled after fitting
        self.scale = None
        self.chi2 = None
        self.dof = None
        self.chi2_red = None
        self.x_plot = None
        self.y_scaled = None

def load_dusty_grid(grid_dir):
    """
    Looks for subfolders named like 'Tstar_4000_Tdust_1100_tau_0_03' containing 'sed.dat'
    with two columns: lam[μm], flux[=λFλ].
    """

    grid_dir = Path(grid_dir)
    rx =  re.compile(r"^Tstar_(\d+)_Tdust_(\d+)_tau_([0-9_]+)_thick_([0-9_]+)$")

    models = []
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
        shell_thickness = float(match.group(4).replace("_", "."))

        table = ascii.read(sed_path, names=["lam", "flux"], comment="#", fast_reader=False)
        lam_um = np.array(table["lam"], float)
        lamFlam = np.array(table["flux"], float)
        models.append(DustyModel(sub, tstar, tdust, tau, shell_thickness, lam_um, lamFlam))

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

        if use_weights and np.any(ey_data > 0):
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
            a = np.sum(weights * y_data * y_mod_on_data) / np.sum(weights * y_mod_on_data**2)
            chi2 = np.sum(weights * (y_data - a*y_mod_on_data)**2)
        else:
            a = np.sum(y_data * y_mod_on_data) / np.sum(y_mod_on_data**2)
            chi2 = np.sum((y_data - a*y_mod_on_data)**2)

        N = len(y_data)
        dof = max(N-1, 0)

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

        
        mask_data = (x_sed >= nu_mod.min()) & (x_sed <= nu_mod.max())

        x_data = x_sed[mask_data]
        y_data = y_sed[mask_data]
        ey_data = ey_sed[mask_data]

        if len(x_data) == 0:
            model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
            model.x_plot, model.y_scaled = nu_mod, Fnu_mod
            return model

        # interpolate model onto observed frequencies
        y_mod_on_data = np.interp(x_data, nu_mod, Fnu_mod)

        if use_weights and np.any(ey_data > 0):
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf)**2
            a = np.sum(weights * y_data * y_mod_on_data) / np.sum(weights * y_mod_on_data**2)
            chi2 = np.sum(weights * (y_data - a*y_mod_on_data)**2)
        else:
            a = np.sum(y_data * y_mod_on_data) / np.sum(y_mod_on_data**2)
            chi2 = np.sum((y_data - a*y_mod_on_data)**2)

        N = len(y_data)
        dof = max(N-1, 0)

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

    oid = sed['oid']

    model_map = {}

    for m in models:
        fit_scale_to_sed(m, sed, y_mode=y_mode, use_weights=use_weights)
        rows.append(dict(
            folder=str(m.folder), oid=oid,
            Tstar=m.Tstar, Tdust=m.Tdust, tau=m.tau,
            scale=m.scale, chi2=m.chi2, dof=m.dof,
            chi2_red=m.chi2_red
        ))
        model_map[str(m.folder)] = m

    df = pd.DataFrame(rows).sort_values("chi2_red")
    df._models = model_map

    return df

def plot_best_fit_dusty_model(df, sed, y_mode="Flam", top_n=10, keep_sed_limits=False,
                              y_padding_frac=0.5, logx=True, logy=True, secax=False, 
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
        if i != 0:
            linestyle = 'dotted'
            alpha=0.6
        else:
            linestyle = '-'
            alpha=1
        m = models[r['folder']]
        ax.plot(m.x_plot, m.y_scaled, lw=2, color=palette[i % len(palette)], linestyle=linestyle, alpha=alpha,
                label=f"T*: {m.Tstar} K, T_dust: {m.Tdust} K, tau: {m.tau}, shell: {m.shell_thickness} | $\\chi^2$={r['chi2']:.2f}, $\\chi^2_\\mathrm{{red}}$={r['chi2_red']:.2f}")

    # SED data
    plot_sed(sed, ax=ax, y_mode=y_mode, logx=logx, logy=logy, secax=secax, savepath=None)

    if keep_sed_limits:
        ymin, ymax = np.nanmin(y_sed), np.nanmax(y_sed)

        lymin = np.log10(ymin)
        lymax = np.log10(ymax)

        lower_decade = np.floor(lymin)
        upper_decade = np.ceil(lymax)

        if np.isclose(lymin, lower_decade):
            lower_decade -= 1
        if np.isclose(lymax, upper_decade):
            upper_decade += 1

        lymin_pad = y_padding_frac * (lymin + lower_decade)
        lymax_pad = y_padding_frac * (lymax + upper_decade)

        ax.set_ylim(10**lymin_pad, 10**lymax_pad)

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

    # these will always be our x ranges
    ax.set_xlim(2e-1, 2e1)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Saved plot to {savepath}")
    else:
        plt.show()