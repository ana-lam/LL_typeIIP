import os
from multiprocessing import get_context

import emcee
import numpy as np
from astropy import constants as const
from pathlib import Path

from ..dusty.runner import DustyRunner
from ..dusty.scaling import compute_chi2, bin_spectrum_to_model_grid
from ..dusty.custom_spectrum import NugentIIPSeries
from ..sed.build import _prepare_sed_xy, _unwrap_sed
from .priors import log_prior
from ..emulator import DustyNNEmulator, DustyTemplateEmulator, DustySingleSpectrumEmulator


def compute_group_weight_scales(lam_um, lamFlam, sed, spec_precomp,
                                err_floor_frac=0.1, equal_weights=True):
    """
    Compute rescaling factors so that W1, W2, and the spectrum each
    contribute equally (1/3) to the total denominator in the analytic
    scale solve -- i.e., group-level equal weighting, not point-level.

    Returns dict: {"W1": scale_W1, "W2": scale_W2, "spectrum": scale_spec}
    such that after applying these per-group multipliers to each group's
    (weight * y_mod^2) sum, all three groups contribute equally.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode="Flam")
    bands = np.array(sed["bands"])
    is_ul = np.array(sed.get("is_ul", np.zeros(len(x_sed), dtype=bool)))

    mask_data = (x_sed >= lam_um.min()) & (x_sed <= lam_um.max())
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
        y_mod_det = np.interp(x_sed[band_mask], lam_um, lamFlam)
        w = 1.0 / np.clip(ey_det, 1e-99, np.inf) ** 2
        den_by_band[band] = np.sum(w * y_mod_det ** 2)

    spec_y_mod = lamFlam[spec_precomp["covered_mask"]]
    den_spec = np.sum(spec_precomp["weight"] * spec_y_mod ** 2)

    # target: each group contributes equally -> scale each group's raw
    # denominator so they all equal the SAME target value (use the mean
    # of the three raw contributions as a neutral target)
    raw = {"W1": den_by_band.get("W1", 0.0),
           "W2": den_by_band.get("W2", 0.0),
           "spectrum": den_spec}

    target = np.mean([v for v in raw.values() if v > 0])

    scales = {}
    for k, v in raw.items():
        scales[k] = (target / v) if v > 0 else 0.0

    return scales, raw

def _ls_scale_and_chi2(lam_um, lamFlam, sed, a=None, y_mode="Flam", use_weights=True,
                    err_floor_frac=0.1, equal_weights=False, quadratic_weights=False,
                    spec_binned_flux=None, spec_weight=None, spec_covered_mask=None, group_weight_scales=None):
    """Analytic best-fit scale and chi2 on the SED grid; if ``a`` is provided, only chi2.

    - If a is None, compute the analytic best-fit scale using detections only.
    - If a is provided from theta (MCMC), use it directly to compute chi2.

    group_weight_scales : dict or None
        {"W1": scale_W1, "W2": scale_W2, "spectrum": scale_spec}. If given,
        each band and the spectrum are rescaled independently so all three
        groups contribute equally (or per whatever ratio you specify) to
        the analytic scale/chi2 solve, rather than photometric points being
        implicitly grouped together vs. spectrum as two groups.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)
    is_ul = np.array(sed.get("is_ul", np.zeros(len(x_sed), dtype=bool)))
    bands = np.array(sed.get("bands", [""] * len(x_sed)))

    if y_mode == "Flam":
        x_mod = lam_um
        y_mod = lamFlam
    else:
        raise NotImplementedError("group_weight_scales currently only supports y_mode='Flam'.")

    mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
    det_mask = mask_data & ~is_ul & np.isfinite(ey_sed) & (ey_sed > 0)
    ul_mask = mask_data & is_ul

    if not np.any(det_mask):
        return np.nan, np.inf

    x_det = x_sed[det_mask]
    y_det = y_sed[det_mask]
    ey_det = ey_sed[det_mask]
    bands_det = bands[det_mask]

    if use_weights and err_floor_frac > 0.0:
        if equal_weights:
            ey_det = err_floor_frac * y_det
        elif quadratic_weights:
            ey_det = np.sqrt(ey_det ** 2 + (err_floor_frac * y_det) ** 2)
        else:
            ey_det = np.maximum(ey_det, err_floor_frac * y_det)

    y_mod_det = np.interp(x_det, x_mod, y_mod)
    w_phot = 1.0 / np.clip(ey_det, 1e-99, np.inf) ** 2

    # per-point group scale (1.0 default if no rebalancing requested)
    if group_weight_scales is not None:
        band_scale = np.array([group_weight_scales.get(b, 1.0) for b in bands_det])
        w_phot = w_phot * band_scale

    spec_y_mod = None
    if spec_covered_mask is not None:
        spec_y_mod = y_mod[spec_covered_mask]
    spec_scale = group_weight_scales.get("spectrum", 1.0) if group_weight_scales else 1.0

    if a is None:
        num = np.sum(w_phot * y_det * y_mod_det)
        den = np.sum(w_phot * y_mod_det ** 2)
        if spec_y_mod is not None:
            num += spec_scale * np.sum(spec_weight * spec_binned_flux * spec_y_mod)
            den += spec_scale * np.sum(spec_weight * spec_y_mod ** 2)
        a = num / den

    chi2 = np.sum(w_phot * (y_det - a * y_mod_det) ** 2)
    if spec_y_mod is not None:
        chi2 += spec_scale * np.sum(spec_weight * (spec_binned_flux - a * spec_y_mod) ** 2)

    if np.any(ul_mask):
        x_ul = x_sed[ul_mask]
        y_ul = y_sed[ul_mask]
        y_mod_ul = a * np.interp(x_ul, x_mod, y_mod)
        ul_sigma = y_ul / 3.0
        excess = y_mod_ul - y_ul
        chi2 += np.sum(np.where(excess > 0, (excess / ul_sigma) ** 2, 0.0))

    return float(a), float(chi2)

def precompute_spectrum_binning(spectrum, lam_ref_um, spectrum_err_floor_frac=0.1):
    """
    Bin an observed spectrum onto a fixed model wavelength grid ONCE, before
    the sampler starts. Returns arrays already restricted to covered bins,
    plus the boolean mask into lam_ref_um selecting those bins -- so every
    subsequent likelihood call just does lamFlam[mask] instead of re-binning.

    lam_ref_um should be the model's native output grid (e.g. DUSTY's 105-pt
    grid, or the emulator's emu.lam_ref) -- the same grid every
    evaluate_model() call returns, regardless of (tdust, tau).
    """
    if spectrum is None:
        return None

    binned_flux, n_pix = bin_spectrum_to_model_grid(
        lam_ref_um, spectrum["lam_um"], spectrum["flux"]
    )
    covered_mask = np.isfinite(binned_flux) & (n_pix > 0)

    spec_binned_flux = binned_flux[covered_mask]
    sigma_spec = spectrum_err_floor_frac * np.clip(np.abs(spec_binned_flux), 1e-99, np.inf)
    spec_weight = 1.0 / sigma_spec ** 2

    return dict(
        covered_mask=covered_mask,
        binned_flux=spec_binned_flux,
        weight=spec_weight,
        n_pix=n_pix[covered_mask],
    )

##############################################
## ------- Likelihood / posterior --------- ##
##############################################

def log_likelihood(theta, sed, dusty_runner,
                   y_mode="Flam", use_weights=True,
                   template=None, template_tag="nugent_iip",
                   tstar_dummy=6000.0, shell_thickness=None,
                   sample_log10a=True, err_floor_frac=0.0, equal_weights=False,
                   quadratic_weights=False, spec_precomp=None, group_weight_scales=None):
    """
    sample_log10a=True: theta includes log10_a as a parameter to sample
    sample_log10a=False: theta has no log10_a; a solved analytically

    Template mode theta (3D): (tdust, log10_tau, log10_a)
    Blackbody mode theta (4D): (tstar, tdust, log10_tau, log10_a)
    """

    sed = _unwrap_sed(sed)
    theta = np.asarray(theta, float)
    a = None

    if sample_log10a:
        if theta.size == 3:
            tdust, log10_tau, log10_a = theta
            a = 10.0**log10_a
            tau = 10.0**log10_tau
            phase = sed.get("phase_days", sed.get("phase", None))
            if phase is None or not np.isfinite(float(phase)):
                return -np.inf
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                tstar=float(tstar_dummy), tdust=float(tdust), tau=float(tau),
                shell_thickness=shell_thickness, template=template,
                phase_days=float(phase), template_tag=str(template_tag),
            )
        elif theta.size == 4:
            tstar, tdust, log10_tau, log10_a = theta
            a = 10.0**log10_a
            tau = 10.0**log10_tau
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                float(tstar), float(tdust), float(tau), shell_thickness=shell_thickness)
        else:
            return -np.inf
    else:
        if theta.size == 2:
            tdust, log10_tau = theta
            tau = 10.0**log10_tau
            phase = sed.get("phase_days", sed.get("phase", None))
            if phase is None or not np.isfinite(float(phase)):
                return -np.inf
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                tstar=float(tstar_dummy), tdust=float(tdust), tau=float(tau),
                shell_thickness=shell_thickness, template=template,
                phase_days=float(phase), template_tag=str(template_tag),
            )
        elif theta.size == 3:
            tstar, tdust, log10_tau = theta
            tau = 10.0**log10_tau
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                float(tstar), float(tdust), float(tau), shell_thickness=shell_thickness)
        else:
            return -np.inf

    if lam_um is None or lamFlam is None:
        return -np.inf

    spec_kwargs = {}
    if spec_precomp is not None:
        spec_kwargs = dict(
            spec_binned_flux=spec_precomp["binned_flux"],
            spec_weight=spec_precomp["weight"],
            spec_covered_mask=spec_precomp["covered_mask"],
        )

    scale, chi2 = _ls_scale_and_chi2(
        lam_um, lamFlam, sed, a=a, y_mode=y_mode, use_weights=use_weights,
        err_floor_frac=err_floor_frac, equal_weights=equal_weights, quadratic_weights=quadratic_weights,
        group_weight_scales=group_weight_scales, **spec_kwargs
    )
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2


def log_probability(theta, sed, dusty_runner, prior_config,
                    y_mode="Flam", use_weights=True,
                    template=None, template_tag="nugent_iip",
                    tstar_dummy=6000.0, shell_thickness=None,
                    sample_log10a=True, err_floor_frac=0.0, equal_weights=False,
                    quadratic_weights=False, spec_precomp=None, group_weight_scales=None):
    lp = log_prior(theta, **prior_config)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, sed, dusty_runner,
                        y_mode=y_mode, use_weights=use_weights,
                        err_floor_frac=err_floor_frac, equal_weights=equal_weights,
                        quadratic_weights=quadratic_weights,
                        template=template, template_tag=template_tag,
                        tstar_dummy=tstar_dummy, shell_thickness=shell_thickness,
                        sample_log10a=sample_log10a, spec_precomp=spec_precomp,
                        group_weight_scales=group_weight_scales)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

##############################################
## --- Multiprocessing + Initialization --- ##
##############################################

def _pick_mp_context(prefer="spawn"):
    """Pick multiprocessing context.
    'fork' for Linux.
    """
    if prefer is None:
        prefer = "spawn"
    if prefer == "fork" and os.name != "nt":
        return get_context("fork")
    return get_context("spawn")

def _initialize_walkers(nwalkers, best, prior_config, ndim, init_mode="hybrid",
                        init_scales=None, max_tries=5000, random_seed=42):
    """
    init_mode:
        - "around_best": gaussian around best
        - "broad": uniform in prior bounds
        - "hybrid": gaussian around best with some fraction uniform in prior bounds
    init_scales keys:
        tstar_frac, tdust_frac (fractional around best)
        log10_tau, log10_a (dex)
    """

    rng = np.random.default_rng(random_seed)

    if init_scales is None:
        init_scales = dict(tstar_frac=0.15, 
                           tdust_frac=0.15, 
                           log10_tau=0.5, 
                           log10_a=0.5)
        
    tstar_bounds = prior_config.get("tstar_bounds", (1000.0, 10000.0))
    tdust_bounds = prior_config.get("tdust_bounds", (100.0, 1750.0))
    log10_tau_bounds = prior_config.get("log10_tau_bounds", (-2.0, 1))
    log10_a_bounds = prior_config.get("log10_a_bounds", (-20.0, 0.0))

    # ---- clip best to prior bounds in case grid best is at an old boundary ----
    best = best.copy()
    best["tdust"] = np.clip(best["tdust"],
                            tdust_bounds[0] + 10.0,
                            tdust_bounds[1] - 10.0)
    best["log10_tau"] = np.clip(best["log10_tau"],
                                log10_tau_bounds[0] + 0.1,
                                log10_tau_bounds[1] - 0.1)
    if "tstar" in best:
        best["tstar"] = np.clip(best["tstar"],
                                tstar_bounds[0] + 10.0,
                                tstar_bounds[1] - 10.0)
    if "log10_a" in best:
        best["log10_a"] = np.clip(best["log10_a"],
                                log10_a_bounds[0] + 0.1,
                                log10_a_bounds[1] - 0.1)

    tstar_bounds = prior_config.get("tstar_bounds", (1000.0, 10000.0))
    tdust_bounds = prior_config.get("tdust_bounds", (100.0, 1750.0))
    log10_tau_bounds = prior_config.get("log10_tau_bounds", (-2.0, 1.0))
    log10_a_bounds = prior_config.get("log10_a_bounds", (-20.0, 0.0))

    def draw_around():
        if ndim == 2: # template, analytic a
            return np.array([
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" not in best: # template, sample log10_a
            return np.array([
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" in best: # blackbody, analytic a
            return np.array([
                best["tstar"] * (1.0 + init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        else: # ndim==4, blackbody, sample log10_a
            return np.array([
                best["tstar"] * (1.0 + init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)

    def draw_broad():
        """
        Use a 3-sigma envelope around best to keep walkers in reasonable territory.
        """
        scale = 3.0 # how many init_scales wide to draw from
        if ndim == 2: # template, analytic a
            cand = np.array([
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" not in best: # template, sample log10_a
            cand = np.array([
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + scale * init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" in best: # blackbody, analytic a
            cand = np.array([
                best["tstar"] * (1. + scale * init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        else: # ndim==4, blackbody, sample log10_a
            cand = np.array([
                best["tstar"] * (1. + scale * init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + scale * init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        # Clip to prior bounds
        if ndim == 2:
            cand[0] = np.clip(cand[0], *tdust_bounds)
            cand[1] = np.clip(cand[1], *log10_tau_bounds)
        elif ndim == 3 and "tstar" not in best:
            cand[0] = np.clip(cand[0], *tdust_bounds)
            cand[1] = np.clip(cand[1], *log10_tau_bounds)
            cand[2] = np.clip(cand[2], *log10_a_bounds)
        elif ndim == 3 and "tstar" in best:
            cand[0] = np.clip(cand[0], *tstar_bounds)
            cand[1] = np.clip(cand[1], *tdust_bounds)
            cand[2] = np.clip(cand[2], *log10_tau_bounds)
        else:
            cand[0] = np.clip(cand[0], *tstar_bounds)
            cand[1] = np.clip(cand[1], *tdust_bounds)
            cand[2] = np.clip(cand[2], *log10_tau_bounds)
            cand[3] = np.clip(cand[3], *log10_a_bounds)
        return cand
    
    # 80/20 split around best
    n_around = int(0.80 * nwalkers)
    
    p0 = np.zeros((nwalkers, ndim), dtype=float)

    for k in range(nwalkers):
        tries = 0
        while True:
            if init_mode == "around_best":
                cand = draw_around()
            elif init_mode == "broad":
                cand = draw_broad()
            elif init_mode == "hybrid":
                cand = draw_around() if (k < n_around) else draw_broad()
            else:
                raise ValueError("init_mode must be one of: 'around_best', 'broad', 'hybrid'.")

            if np.isfinite(log_prior(cand, **prior_config)):
                p0[k] = cand
                break

            tries += 1
            if tries >= max_tries:
                raise RuntimeError(
                        f"Could not find valid initial position for walker {k}. Try broader bounds or looser priors."
                    )
    
    return p0

##############################################
## -------------- MCMC driver ------------- ##
##############################################

def run_mcmc_for_sed(sed, grid_df, dusty_file_dir, workdir,
                     dust_type="silicate", shell_thickness=2.0,
                     template_path=None, template_tag="nugent_iip",
                     template_kind="auto", spectrum_z=0.0,
                     tstar_dummy=6000.0,
                     nwalkers=32, nsteps=4000, burn_in=1000, y_mode="Flam",
                     use_weights=True, n_cores=4, mp_prefer="fork",
                     random_seed=42, posterior_mode="data", mix_weight=0.3,
                     tdust_sigma_frac=0.2, log10_tau_sigma=0.3, log10_a_sigma=0.5,
                     init_mode="hybrid", init_scales=None, progress_every=None,
                     cache_dir=None, cache_ndigits=4, cache_max=5000, use_tmp=True,
                     run_tag=None, evaluator_mode="dusty", emulator_path=None,
                     sample_log10a=True, log10_a_range=3.0, err_floor_frac=0.0, 
                     equal_weights=False, quadratic_weights=False, 
                     spectrum=None, spectrum_err_floor_frac=0.1,
                     spectrum_weight_scale=1.0):
    """
    Run emcee for a given SED using DUSTY models.

    Always samples a (via log10_a).
    Use posterior_mode:
      - "data": bounds-only posterior
      - "anchored": regularized around grid best
      - "mixture": soft hint from grid + broad exploration

    Template mode:
      - needs sed['phase_days']
      - theta = (tdust, log10_tau, log10_a)  [ndim=3]
    """

    sed = _unwrap_sed(sed)

    if cache_dir is None and template_path is not None:
        cache_dir = str(Path(workdir) / "dusty_npz_cache_template")
    elif cache_dir is None:
        cache_dir = str(Path(workdir) / "dusty_npz_cache_blackbody")

    template = None
    template_mode = (template_path is not None)
    if template_mode:
        resolved_kind = template_kind
        if resolved_kind == "auto":
            resolved_kind = "single_spectrum" if str(template_path).lower().endswith(".flm") else "series"

        if resolved_kind == "single_spectrum":
            template = NugentIIPSeries.from_single_spectrum(
                str(Path(template_path).resolve()), z=spectrum_z
            )
        else:
            template = NugentIIPSeries(str(Path(template_path).resolve()))

    # Initialize the emulator if specified
    if evaluator_mode == "emulator":
        if emulator_path is None:
            raise ValueError("emulator_path must be specified when using emulator mode.")
        if not Path(emulator_path).exists():
            raise FileNotFoundError(f"Emulator file not found: {emulator_path}")
        if template_mode:
            resolved_kind = template_kind
            if resolved_kind == "auto":
                resolved_kind = "single_spectrum" if str(template_path).lower().endswith(".flm") else "series"
            if resolved_kind == "single_spectrum":
                dusty_runner = DustySingleSpectrumEmulator(emulator_path)
            else:
                dusty_runner = DustyTemplateEmulator(emulator_path)
        else:
            dusty_runner = DustyEmulator(emulator_path)
    
        print(f"[evaluator] mode=EMULATOR loaded from {emulator_path}")


    elif evaluator_mode == "dusty":
        print(f"[evaluator] mode=DUSTY cache={cache_dir}")

        dusty_runner = DustyRunner(
                            base_workdir=workdir,
                            dusty_file_dir=dusty_file_dir,
                            dust_type=dust_type,
                            shell_thickness=shell_thickness,
                            cache_dir=cache_dir,
                            cache_ndigits=cache_ndigits,
                            cache_max=cache_max,
                            use_tmp=use_tmp,
                            run_tag=run_tag
                        )
    else:
        raise ValueError(f"evaluator_mode must be either 'emulator' or 'dusty'")
    
    # best-fit from grid
    grid_df = grid_df.sort_values("chi2_red").reset_index(drop=True)
    best_row = grid_df.iloc[0]

    if template_mode:
        ndim = 3 if sample_log10a else 2
        best = {
            "tdust": float(best_row["tdust"]),
            "log10_tau": np.log10(float(best_row["tau"])),
            "log10_a": np.log10(float(best_row["scale"])),
        }
        log10_a_best = best["log10_a"]
        log10_a_bounds = (
            (log10_a_best - log10_a_range, log10_a_best + log10_a_range)
            if sample_log10a else
            (-20.0, 0.0)
        )
        prior_config = dict(
            tdust_bounds=(50., 1750.),
            log10_tau_bounds=(-2., 1.),
            log10_a_bounds=log10_a_bounds,
            best=best,
            prior_mode=posterior_mode,
            mix_weight=mix_weight,
            tdust_sigma_frac=tdust_sigma_frac,
            log10_tau_sigma=log10_tau_sigma,
            log10_a_sigma=log10_a_sigma,
            sample_log10a=sample_log10a,
        )
    else:
        ndim = 4 if sample_log10a else 3
        best = {
            "tstar": float(best_row["tstar"]),
            "tdust": float(best_row["tdust"]),
            "log10_tau": np.log10(float(best_row["tau"])),
            "log10_a": np.log10(float(best_row["scale"])),
        }
        log10_a_best = best["log10_a"]
        log10_a_bounds = (
            (log10_a_best - log10_a_range, log10_a_best + log10_a_range)
            if sample_log10a else (-20., 0.)
        )
        prior_config = dict(
            tstar_bounds=(2000., 12000.),
            tdust_bounds=(100., 1750.),
            log10_tau_bounds=(-2., 1.),
            log10_a_bounds=log10_a_bounds,
            best=best,
            prior_mode=posterior_mode,
            mix_weight=mix_weight,
            tstar_sigma_frac=0.3,
            tdust_sigma_frac=tdust_sigma_frac,
            log10_tau_sigma=log10_tau_sigma,
            log10_a_sigma=log10_a_sigma,
            sample_log10a=sample_log10a,
        )


    # # initial positions from best
    # ndim = 4 # tstar, tdust, log10_tau, log10_a
    # p0 = np.zeros((nwalkers, ndim))
    # p0[:, 0] = best["tstar"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    # p0[:, 1] = best["tdust"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    # p0[:, 2] = best["log10_tau"] + 0.2 * np.random.randn(nwalkers)
    # p0[:, 3] = best["log10_a"] + 0.2 * np.random.randn(nwalkers)

    p0 = _initialize_walkers(
        nwalkers=nwalkers,
        best=best,
        prior_config=prior_config,
        ndim=ndim,
        init_mode=init_mode,
        init_scales=init_scales,
        random_seed=random_seed
    )

    spec_precomp = None
    if spectrum is not None:
        probe_tdust = best["tdust"]
        probe_tau = 10.0 ** best["log10_tau"]
        probe_phase = sed.get("phase_days", sed.get("phase", None))
        lam_probe, lamFlam_probe, _ = dusty_runner.evaluate_model(
            tstar=float(tstar_dummy), tdust=float(probe_tdust), tau=float(probe_tau),
            shell_thickness=shell_thickness, template=template,
            phase_days=float(probe_phase) if probe_phase is not None else None,
            template_tag=template_tag,
        )
        if lam_probe is None:
            raise RuntimeError("Could not evaluate model at grid-best point to "
                              "determine wavelength grid for spectrum binning.")
        spec_precomp = precompute_spectrum_binning(
            spectrum, lam_probe, spectrum_err_floor_frac=spectrum_err_floor_frac
        )
        print(f"[spectrum] {spec_precomp['covered_mask'].sum()} model wavelength "
              f"bins covered by spectrum (out of {len(lam_probe)} total).")

    # for rebalancing the contributions of W1, W2, and spectrum to the analytic scale/chi2 solve
    group_scales = None
    if spec_precomp is not None:
        group_scales, raw_contributions = compute_group_weight_scales(
            lam_probe, lamFlam_probe, sed, spec_precomp,
            err_floor_frac=err_floor_frac, equal_weights=equal_weights
        )
        print(f"Raw contributions: {raw_contributions}")
        print(f"Rebalancing scales for 1/3-1/3-1/3: {group_scales}")

    if burn_in >= nsteps:
        raise ValueError("burn_in must be less than nsteps.")
    
    ctx = _pick_mp_context(prefer=mp_prefer)
    ncores = min(int(n_cores), nwalkers)

    sampler_kwargs = dict(moves=emcee.moves.StretchMove(a=2.0))

    def _run_sampler(sampler):
        print(f"Running MCMC: ndim={ndim}, nwalkers={nwalkers}, "
          f"nsteps={nsteps}...", flush=True)
        if progress_every is not None:
            state = p0
            for i, state in enumerate(
                    sampler.sample(state, iterations=nsteps), start=1):
                if (i == 1) or (i % progress_every == 0) or (i == nsteps):
                    print(f"[emcee] step {i}/{nsteps}", flush=True)
        else:
            sampler.run_mcmc(p0, nsteps, progress=True)

    if evaluator_mode == "emulator":
        # no pool
        print(f"[pool] emulator mode — running single-threaded (no pool)", flush=True)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(sed, dusty_runner, prior_config, y_mode, use_weights,
                template, template_tag, tstar_dummy, shell_thickness, sample_log10a, err_floor_frac, equal_weights, quadratic_weights, spec_precomp, group_scales),
            **sampler_kwargs
        )
        _run_sampler(sampler)
    else:
        print(f"[pool] dusty mode — using {ncores} cores", flush=True)
        with ctx.Pool(processes=ncores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, 
                args=(sed, dusty_runner, prior_config, y_mode, use_weights,
                      template, template_tag, tstar_dummy, shell_thickness, sample_log10a, err_floor_frac, equal_weights, quadratic_weights, spec_precomp, group_scales),
                pool=pool,
                **sampler_kwargs,
            )
            _run_sampler(sampler)

    # discard burn-in and flatten
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burn_in, flat=True)

    # for diagnostics
    chain_full = sampler.get_chain() 
    logp_full  = sampler.get_log_prob() 
    af = sampler.acceptance_fraction

    try:
        tau_int = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau_int = None


    # unpack
    if sample_log10a:
        if ndim == 3: # template + sample_log10a
            tdust_samples = samples[:, 0]
            log10_tau_samples = samples[:, 1]
            log10_a_samples = samples[:, 2]
            tstar_samples = np.full_like(tdust_samples, float(tstar_dummy))
        else: # blackbody + sample_log10a
            tstar_samples = samples[:, 0]
            tdust_samples = samples[:, 1]
            log10_tau_samples = samples[:, 2]
            log10_a_samples = samples[:, 3]
    else:
        print("Deriving log10_a analytically for each posterior sample...", flush=True)
        if ndim == 2: # template + analytic a
            tdust_samples = samples[:, 0]
            log10_tau_samples = samples[:, 1]
            tstar_samples = np.full_like(tdust_samples, float(tstar_dummy))
        else: # blackbody + analytic a
            tstar_samples = samples[:, 0]
            tdust_samples = samples[:, 1]
            log10_tau_samples = samples[:, 2]

        phase = sed.get("phase_days", sed.get("phase", None))
        log10_a_samples = np.full(len(samples), np.nan)

        for i, (tdust_i, log10_tau_i) in enumerate(zip(tdust_samples, log10_tau_samples)):
            if i % 500 == 0 or i == len(samples) - 1:
                print(f"  Processing sample {i+1}/{len(samples)}...", flush=True)
            tau_i = 10.0 ** log10_tau_i
            tstar_i = float(tstar_samples[i])
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                tstar=tstar_i,
                tdust=tdust_i,
                tau=tau_i,
                shell_thickness=shell_thickness,
                template=template,
                phase_days=float(phase) if phase is not None else None,
                template_tag=template_tag,
            )
            if lam_um is not None:
                a_i, _ = _ls_scale_and_chi2(
                lam_um, lamFlam, sed, a=None, y_mode=y_mode, use_weights=use_weights,
                err_floor_frac=err_floor_frac, equal_weights=equal_weights, quadratic_weights=quadratic_weights,
                group_weight_scales=group_scales,
                **({} if spec_precomp is None else dict(
                    spec_binned_flux=spec_precomp["binned_flux"],
                    spec_weight=spec_precomp["weight"],
                    spec_covered_mask=spec_precomp["covered_mask"],
                ))
                )
                if np.isfinite(a_i) and (a_i > 0):
                    log10_a_samples[i] = np.log10(a_i)
        print(f"  Derived log10_a for {np.sum(np.isfinite(log10_a_samples))}"
                f"/{len(log10_a_samples)} samples.", flush=True)

    tau_samples = 10.0 ** log10_tau_samples
    a_samples   = 10.0 ** np.where(np.isfinite(log10_a_samples),
                                    log10_a_samples, np.nan)

    results = dict(
        samples=samples,
        log_prob=log_prob_samples,
        tstar=tstar_samples,
        tdust=tdust_samples,
        tau=tau_samples,
        a=a_samples,
        grid_best=best,
        prior_config=prior_config,
        sampler=sampler,
        chain=chain_full,
        log_prob_chain=logp_full,
        acceptance_fraction=af,
        autocorr_time=tau_int,
        template_mode=template_mode,
        template_tag=template_tag,
        log10_a_samples=log10_a_samples,
        analytic_log10a=not sample_log10a,
        err_floor_frac=err_floor_frac,
        equal_weights=equal_weights,
        quadratic_weights=quadratic_weights,
        spectrum=spectrum,
        spectrum_err_floor_frac=spectrum_err_floor_frac,
        group_weight_scales=group_scales,
    )

    return results