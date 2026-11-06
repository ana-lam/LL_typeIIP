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


def bin_spectrum_to_model_grid(lam_model_um, lam_spec_um, flux_spec):
    """
    Bins spectrum onto DUSTY's coarser model grid,
    averaging over +/- local_bin_width/2 around each model wavelength point
    """

    lam_model_um = np.asarray(lam_model_um, dtype=float)
    lam_spec_um = np.asarray(lam_spec_um, dtype=float)
    flux_spec = np.asarray(flux_spec, dtype=float)

    order = np.argsort(lam_model_um)
    lam_sorted = lam_model_um[order]

    bin_edges = np.empty(len(lam_sorted) + 1)
    bin_edges[1:-1] = 0.5 * (lam_sorted[:-1] + lam_sorted[1:])
    bin_edges[0] = lam_sorted[0] - (bin_edges[1] - lam_sorted[0])
    bin_edges[-1] = lam_sorted[-1] + (lam_sorted[-1] - bin_edges[-2])

    binned_flux_sorted = np.full(len(lam_sorted), np.nan)
    n_pix_sorted = np.zeros(len(lam_sorted), dtype=int)

    for i in range(len(lam_sorted)):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (lam_spec_um >= lo) & (lam_spec_um < hi) & np.isfinite(flux_spec)
        if np.any(mask):
            binned_flux_sorted[i] = np.mean(flux_spec[mask])
            n_pix_sorted[i] = np.sum(mask)

    binned_flux = np.empty_like(binned_flux_sorted)
    n_pix = np.empty_like(n_pix_sorted)
    binned_flux[order] = binned_flux_sorted
    n_pix[order] = n_pix_sorted

    return binned_flux, n_pix


def compute_group_weight_scales(x_mod, y_mod, sed, spec_y_obs, spec_y_mod, spec_weight,
                                err_floor_frac=0.1, equal_weights=True,
                                target_fractions=None):
    """
    Compute rescaling factors so that W1, W2, and the spectrum contribute
    according to target_fractions (default: equal 1/3 each) to the total
    denominator in the analytic scale solve.

    target_fractions : dict or None
        e.g. {"W1": 0.45, "W2": 0.45, "spectrum": 0.1} to deliberately
        down-weight the spectrum's influence relative to strict equal
        weighting. Must sum to 1.0 (not strictly enforced, but should).
        Default None -> equal weighting (1/3 each).
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode="Flam")
    bands = np.array(sed["bands"])
    is_ul = np.array(sed.get("is_ul", np.zeros(len(x_sed), dtype=bool)))

    mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
    det_mask = mask_data & ~is_ul & np.isfinite(ey_sed) & (ey_sed > 0)

    den_by_band = {}
    for band in ["W1", "W2"]:
        band_mask = det_mask & (bands == band)
        if not np.any(band_mask):
            den_by_band[band] = 0.0
            continue
        y_det = y_sed[band_mask]
        ey_det = ey_sed[band_mask]
        if equal_weights:
            ey_det = err_floor_frac * y_det
        else:
            ey_det = np.maximum(ey_det, err_floor_frac * y_det)
        y_mod_det = np.interp(x_sed[band_mask], x_mod, y_mod)
        w = 1.0 / np.clip(ey_det, 1e-99, np.inf) ** 2
        den_by_band[band] = np.sum(w * y_mod_det ** 2)

    den_spec = np.sum(spec_weight * spec_y_mod ** 2) if spec_y_mod is not None else 0.0

    raw = {"W1": den_by_band.get("W1", 0.0),
           "W2": den_by_band.get("W2", 0.0),
           "spectrum": den_spec}

    if target_fractions is None:
        target_fractions = {"W1": 1/3, "W2": 1/3, "spectrum": 1/3}

    # total denominator we want AFTER rescaling: pick an arbitrary reference
    # (sum of raw contributions), then allocate that total according to
    # target_fractions rather than forcing every group to the same value
    total_raw = sum(v for v in raw.values() if v > 0)

    scales = {}
    for k, v in raw.items():
        target_v = target_fractions.get(k, 0.0) * total_raw
        scales[k] = (target_v / v) if v > 0 else 0.0

    return scales, raw


def fit_scale_to_sed(model, sed, y_mode="Flam", use_weights=True, spectrum=None,
                    equal_weights=False, quadratic_weights=False, err_floor_frac=0.1,
                    spectrum_weight_mode="err_floor", spectrum_err_floor_frac=0.1,
                    equal_group_weights=True, target_fractions=None):
    """
    Find best-fit scale factor for DUSTY model to match SED using least-squares.

    spectrum : dict or None
        Optional. {"lam_um": array, "flux": array} of an observed spectrum,
        de-redshifted to rest frame, in the same units as y_mode ('Flam' only
        -- Fnu not currently supported for spectral fitting). If given, the
        spectrum is binned onto the model's native wavelength grid (averaging
        within each grid cell's local spacing) and included as additional
        residual terms in the same weighted least-squares scale/chi2 solve
        used for the photometric SED points.
    spectrum_weight_mode : str
        'err_floor' -- use spectrum_err_floor_frac * |binned_flux| as the
        per-bin sigma (simple, robust default when no reliable per-pixel
        error array is propagated through binning).
    spectrum_err_floor_frac : float
        Fractional error floor applied per spectral bin when
        spectrum_weight_mode='err_floor'.
    equal_group_weights : bool
        If True (default) and spectrum is given, W1, W2, and the spectrum
        are automatically rescaled to contribute equally (1/3 each) to the
        scale/chi2 solve via compute_group_weight_scales(), rather than
        relying on relative-error weighting alone (which gives disproportionate
        leverage to whichever dataset has the largest raw model/obs ratio).
        If False, falls back to plain per-point relative-error weighting with
        no group-level rebalancing (spectrum and photometry combine with
        whatever relative weight the per-point errors happen to produce).
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)
    is_ul = np.array(sed.get("is_ul", [False] * len(sed["bands"])), dtype=bool)
    bands = np.array(sed.get("bands", [""] * len(x_sed)))

    if y_mode == "Flam":
        x_mod = model.lam_um
        y_mod = model.lamFlam
    elif y_mode == "Fnu":
        if spectrum is not None:
            raise NotImplementedError("Spectral fitting currently only supports y_mode='Flam'.")
        lam_cm = model.lam_um * 1e-4
        nu_mod = const.c.cgs.value / lam_cm
        Fnu_mod = model.lamFlam * (lam_cm / const.c.cgs.value)
        order = np.argsort(nu_mod)
        x_mod = nu_mod[order]
        y_mod = Fnu_mod[order]
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")

    # overlap
    mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
    x_data  = x_sed[mask_data]
    y_data  = y_sed[mask_data]
    ey_data = ey_sed[mask_data]
    ul_mask = is_ul[mask_data]
    bands_data = bands[mask_data]

    if len(x_data) == 0:
        model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
        model.x_plot, model.y_scaled = x_mod, y_mod
        model.n_ul_active = 0
        return model

    y_mod_on_data = np.interp(x_data, x_mod, y_mod)

    # --- scale from detections only ---
    det = ~ul_mask
    x_det, y_det, ey_det = x_data[det], y_data[det], ey_data[det]
    y_mod_det = y_mod_on_data[det]
    bands_det = bands_data[det]

    if len(y_det) == 0:
        model.scale, model.chi2, model.dof, model.chi2_red = np.nan, np.inf, 0, np.inf
        model.x_plot, model.y_scaled = x_mod, y_mod
        model.n_ul_active = 0
        return model

    # --- bin spectrum onto model grid, if provided ---
    spec_y_mod = None
    spec_y_obs = None
    spec_weight = None
    if spectrum is not None:
        if y_mode != "Flam":
            raise ValueError("spectrum fitting only supported for y_mode='Flam'")
        binned_flux, n_pix = bin_spectrum_to_model_grid(
            x_mod, spectrum["lam_um"], spectrum["flux"])
        covered = np.isfinite(binned_flux) & (n_pix > 0)

        spec_y_obs = binned_flux[covered]
        spec_y_mod = y_mod[covered]

        if spectrum_weight_mode == "err_floor":
            sigma_spec = spectrum_err_floor_frac * np.clip(np.abs(spec_y_obs), 1e-99, np.inf)
        else:
            raise ValueError(f"Unknown spectrum_weight_mode: {spectrum_weight_mode}")

        spec_weight = 1.0 / sigma_spec ** 2

    if use_weights and err_floor_frac > 0.0:
        if equal_weights:
            # ignore real errors entirely; every point weighted purely by its own flux
            ey_det = err_floor_frac * y_det
        elif quadratic_weights:
            # combine REAL reported error with the floor in quadrature
            ey_det = np.sqrt(ey_det**2 + (err_floor_frac * y_det)**2)
        else:
            ey_det = ey_det

    if use_weights and np.any(ey_det > 0):
        w_phot = 1.0 / np.clip(ey_det, 1e-99, np.inf) ** 2
    else:
        w_phot = np.ones_like(y_det)

    # --- group-level equal weighting (W1, W2, spectrum each 1/3) ---
    group_scales = None
    raw_contributions = None
    if spectrum is not None and equal_group_weights:
        group_scales, raw_contributions = compute_group_weight_scales(
            x_mod, y_mod, sed, spec_y_obs, spec_y_mod, spec_weight,
            err_floor_frac=spectrum_err_floor_frac, equal_weights=use_weights,
            target_fractions=target_fractions,
        )
        band_scale = np.array([group_scales.get(b, 1.0) for b in bands_det])
        w_phot = w_phot * band_scale

    spec_scale = group_scales.get("spectrum", 1.0) if group_scales else 1.0

    num = np.sum(w_phot * y_det * y_mod_det)
    den = np.sum(w_phot * y_mod_det ** 2)

    if spec_y_mod is not None:
        num += spec_scale * np.sum(spec_weight * spec_y_obs * spec_y_mod)
        den += spec_scale * np.sum(spec_weight * spec_y_mod ** 2)

    a = num / den

    chi2_det = np.sum(w_phot * (y_det - a * y_mod_det) ** 2)
    n_spec_used = 0
    chi2_spec = 0.0

    if spec_y_mod is not None:
        chi2_spec = spec_scale * np.sum(spec_weight * (spec_y_obs - a * spec_y_mod) ** 2)
        n_spec_used = len(spec_y_obs)

    # UL penalities
    y_mod_ul = a * y_mod_on_data[ul_mask]
    y_ul = y_data[ul_mask]

    sigma_ul = y_ul / 3.0  # recover error

    exceeded = y_mod_ul > y_ul
    excess = y_mod_ul - y_ul
    chi2_ul = np.sum((excess[exceeded] / sigma_ul[exceeded]) ** 2)
    n_ul_active = int(np.sum(exceeded))

    chi2 = chi2_det + chi2_ul + chi2_spec
    dof = max(len(y_det) + n_ul_active + n_spec_used - 1, 0)

    model.scale = float(a)
    model.chi2 = float(chi2)
    model.dof = int(dof)
    model.chi2_red = float(chi2 / dof) if dof > 0 else np.inf
    model.n_ul_active = n_ul_active
    model.n_spec_used = n_spec_used
    model.group_weight_scales = group_scales
    model.group_raw_contributions = raw_contributions
    model.x_plot = x_mod
    model.y_scaled = a * y_mod

    return model


def fit_grid_to_sed(grid_csv, sed, shell_thickness=None, y_mode="Flam", use_weights=True, quadratic_weights=False,
                    template_tag=None, top_k=None, max_tstar=6000.0, spectrum=None,
                    spectrum_err_floor_frac=0.1, equal_group_weights=True, target_fractions=None):
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
    spectrum : dict or None
        Optional observed spectrum to include alongside photometry -- see
        fit_scale_to_sed for details.
    equal_group_weights : bool
        If True (default), W1/W2/spectrum are rebalanced to contribute
        equally to each model's fit. See fit_scale_to_sed.

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

        if template_tag is None:
            # infer tag from the CSV rather than hardcoding
            import pandas as _pd
            _df_peek = _pd.read_csv(grid_csv, nrows=1)
            if 'template_tag' in _df_peek.columns:
                template_tag = str(_df_peek['template_tag'].iloc[0])
            elif 'nugent' in grid_csv.name:
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

    printed_group_scales = False

    equal_weights=False
    if use_weights is False:
        use_weights=True
        equal_weights=True

    for m in models:
        fit_scale_to_sed(
            m, sed, y_mode=y_mode, use_weights=use_weights, 
            equal_weights=equal_weights, quadratic_weights=quadratic_weights,
            err_floor_frac=0.1,
            spectrum=spectrum, spectrum_err_floor_frac=spectrum_err_floor_frac,
            equal_group_weights=equal_group_weights,
            target_fractions=target_fractions,
        )

        if spectrum is not None and not printed_group_scales and getattr(m, "group_weight_scales", None):
            print(f"[group weights] raw denominator contributions: {m.group_raw_contributions}")
            print(f"[group weights] rebalancing scales for equal (1/3-1/3-1/3) weighting: {m.group_weight_scales}")
            printed_group_scales = True

        row = dict(
            folder=str(m.folder),
            oid=oid_for_output,
            tdust=m.Tdust,
            tau=m.tau,
            shell_thickness=m.shell_thickness,
            scale=m.scale,
            chi2=m.chi2,
            dof=m.dof,
            chi2_red=m.chi2_red,
            n_ul_active=getattr(m, 'n_ul_active', 0),
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

        row["n_spec_used"] = getattr(m, "n_spec_used", 0)

        rows.append(row)
        model_map[str(m.folder)] = m

    # create DataFrame and sort by chi2_red
    df = pd.DataFrame(rows).sort_values("chi2_red").reset_index(drop=True)

    df.attrs["models"] = model_map

    best_model_folder = df.iloc[0]['folder']
    best_model = df.attrs["models"][best_model_folder]
    df_resid = sed_residuals(best_model, sed, y_mode=y_mode, print_table=True)
    df.attrs["best_model_residuals"] = df_resid

    return df


def sed_residuals(model, sed, y_mode="Flam", print_table=True):
    """Compute per-band residuals for a fitted model and return a DataFrame."""
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)
    is_ul = np.array(sed.get("is_ul", [False] * len(sed["bands"])), dtype=bool)
    bands = np.array(sed["bands"])

    if y_mode in ["Flam", "Fnu"]:
        x_mod = model.x_plot
        y_mod = model.y_scaled
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")

    y_mod_interp = np.interp(x_sed, x_mod, y_mod)

    rows = []
    chi2_total = 0.0

    for b, x, yo, ym, e, ul in zip(
        bands, x_sed, y_sed, y_mod_interp, ey_sed, is_ul
    ):
        if ul:
            sigma = yo / 3.0
            excess = ym - yo

            if excess > 0:
                residual_sigma = excess / sigma
                chi2_i = residual_sigma**2
            else:
                residual_sigma = 0.0
                chi2_i = 0.0

            err_used = sigma
            residual = yo - ym

        else:
            err_used = e
            residual = yo - ym
            residual_sigma = residual / e
            chi2_i = residual_sigma**2

        chi2_total += chi2_i

        rows.append(
            {
                "band": b,
                "x": x,
                "y_obs": yo,
                "y_model": ym,
                "y_err": e,
                "err_used": err_used,
                "residual": residual,
                "residual_sigma": residual_sigma,
                "chi2_band": chi2_i,
                "is_ul": bool(ul),
                "model_folder": str(model.folder),
                "scale": getattr(model, "scale", np.nan),
                "chi2_total_model": getattr(model, "chi2", np.nan),
                "dof_model": getattr(model, "dof", np.nan),
                "chi2_red_model": getattr(model, "chi2_red", np.nan),
            }
        )

    df_resid = pd.DataFrame(rows)

    if print_table:
        print(f"\nPer-band residuals for {model.folder}:")
        print(
            f"{'Band':<10} {'Obs':>12} {'Model':>12} "
            f"{'Err':>12} {'Residual':>12} {'chi2':>10} {'UL':>5}"
        )
        print("-" * 75)

        for _, row in df_resid.iterrows():
            if row["is_ul"]:
                print(
                    f"{row['band']:<10} "
                    f"{row['y_obs']:>12.4e} "
                    f"{row['y_model']:>12.4e} "
                    f"{'UL':>12} "
                    f"{row['residual']:>+12.4e} "
                    f"{row['chi2_band']:>10.3f}  yes"
                )
            else:
                print(
                    f"{row['band']:<10} "
                    f"{row['y_obs']:>12.4e} "
                    f"{row['y_model']:>12.4e} "
                    f"{row['y_err']:>12.4e} "
                    f"{row['residual_sigma']:>+12.4f} "
                    f"{row['chi2_band']:>10.3f}   no"
                )

        dof = getattr(model, "dof", 0)
        chi2_red = chi2_total / dof if dof > 0 else np.inf
        print(f"\nTotal chi2={chi2_total:.3f}, dof={dof}, chi2_red={chi2_red:.3f}")

    return df_resid

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
#         'Flam' or 'Fnu' — determines whether we compare in lambda F_lambda or F_nu space.
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