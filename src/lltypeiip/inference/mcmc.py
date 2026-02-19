import os
from multiprocessing import get_context

import emcee
import numpy as np
from astropy import constants as const
from pathlib import Path

from ..dusty.runner import DustyRunner
from ..dusty.scaling import compute_chi2
from ..dusty.custom_spectrum import NugentIIPSeries
from ..sed.build import _prepare_sed_xy
from .priors import log_prior


def _unwrap_sed(obj):
    """Unwrap SED dict"""
    if isinstance(obj, dict) and "sed" in obj and isinstance(obj["sed"], dict):
        sed = obj["sed"]
        if "phase_days" not in sed and "phase" in obj:
            sed["phase_days"] = obj["phase"]
        return sed
    return obj

def _ls_scale_and_chi2(lam_um, lamFlam, sed, a=None, y_mode="Flam", use_weights=True):
    """Analytic best-fit scale and chi2 on the SED grid; if ``a`` is provided, only chi2."""
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)

    if y_mode == "Flam":
        x_mod = lam_um
        y_mod = lamFlam
    elif y_mode == "Fnu":
        lam_cm = lam_um * 1e-4
        nu_mod = const.c.cgs.value / lam_cm
        Fnu_mod = lamFlam * (lam_cm / const.c.cgs.value)
        order = np.argsort(nu_mod)
        x_mod = nu_mod[order]
        y_mod = Fnu_mod[order]
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")

    mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
    if not np.any(mask_data):
        return np.nan, np.inf

    x_data = x_sed[mask_data]
    y_data = y_sed[mask_data]
    ey_data = ey_sed[mask_data]

    y_mod_on_data = np.interp(x_data, x_mod, y_mod)

    if a is None:
        if use_weights and np.any(ey_data > 0):
            weights = 1.0 / np.clip(ey_data, 1e-99, np.inf) ** 2
            a = np.sum(weights * y_data * y_mod_on_data) / np.sum(weights * y_mod_on_data ** 2)
            chi2 = np.sum(weights * (y_data - a * y_mod_on_data) ** 2)
        else:
            a = np.sum(y_data * y_mod_on_data) / np.sum(y_mod_on_data ** 2)
            chi2 = np.sum((y_data - a * y_mod_on_data) ** 2)
    else:
        chi2 = compute_chi2(lam_um, lamFlam, sed, a, y_mode=y_mode, use_weights=use_weights)[1]

    return float(a), float(chi2)

##############################################
## ------- Likelihood / posterior --------- ##
##############################################

def log_likelihood(theta, sed, dusty_runner,
                   y_mode="Flam", use_weights=True,
                   template=None, template_tag="nugent_iip",
                   tstar_dummy=6000.0, shell_thickness=None):
    """
    Template mode theta (3D): (tdust, log10_tau, log10_a)
    Blackbody mode theta (4D): (tstar, tdust, log10_tau, log10_a)
    """
    # if len(theta) != 4:
    #     tstar, tdust, log10_tau = theta
    #     tau = 10.0**log10_tau
    #     a = None
    # else:
    #     tstar, tdust, log10_tau, log10_a = theta
    #     tau = 10.0**log10_tau
    #     a = 10.0**log10_a

    sed = _unwrap_sed(sed)

    theta = np.asarray(theta, float)

    if theta.size == 3:
        tdust, log10_tau, log10_a = theta
        tau = 10.0**log10_tau
        a = 10.0**log10_a

        # template needs phase_days
        phase = sed.get("phase_days", sed.get("phase", None))
        if phase is None or not np.isfinite(float(phase)):
            return -np.inf

        lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
            tstar=float(tstar_dummy),
            tdust=float(tdust),
            tau=float(tau),
            shell_thickness=shell_thickness,
            template=template,
            phase_days=float(phase),
            template_tag=str(template_tag),
        )

    elif theta.size == 4:
        tstar, tdust, log10_tau, log10_a = theta
        tau = 10.0**log10_tau
        a = 10.0**log10_a
        lam_um, lamFlam, r1 = dusty_runner.evaluate_model(float(tstar), float(tdust), float(tau), shell_thickness=shell_thickness)

    else:
        return -np.inf


    if lam_um is None or lamFlam is None:
        return -np.inf

    scale, chi2 = _ls_scale_and_chi2(
        lam_um, lamFlam, sed, a=a, y_mode=y_mode, use_weights=use_weights
    )
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2


def log_probability(theta, sed, dusty_runner, prior_config,
                    y_mode="Flam", use_weights=True,
                    template=None, template_tag="nugent_iip",
                    tstar_dummy=6000.0, shell_thickness=None):
    lp = log_prior(theta, **prior_config)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, sed, dusty_runner,
                        y_mode=y_mode, use_weights=use_weights,
                        template=template, template_tag=template_tag,
                        tstar_dummy=tstar_dummy, shell_thickness=shell_thickness)
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
        init_scales = dict(tstar_frac=0.25, 
                           tdust_frac=0.25, 
                           log10_tau=0.8, 
                           log10_a=0.8)

    tstar_bounds = prior_config.get("tstar_bounds", (1000.0, 12000.0))
    tdust_bounds = prior_config.get("tdust_bounds", (100.0, 1500.0))
    log10_tau_bounds = prior_config.get("log10_tau_bounds", (-4.0, 2.0))
    log10_a_bounds = prior_config.get("log10_a_bounds", (-30.0, 30.0))

    def draw_around():
        if ndim == 3:
            return np.array([
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        else:
            return np.array([
                best["tstar"] * (1.0 + init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)

    def draw_broad():
        if ndim == 3:
            return np.array([
                rng.uniform(*tdust_bounds),
                rng.uniform(*log10_tau_bounds),
                rng.uniform(*log10_a_bounds),
            ], dtype=float)
        else:
            return np.array([
                rng.uniform(*tstar_bounds),
                rng.uniform(*tdust_bounds),
                rng.uniform(*log10_tau_bounds),
                rng.uniform(*log10_a_bounds),
            ], dtype=float)
    
    p0 = np.zeros((nwalkers, ndim), dtype=float)

    for k in range(nwalkers):
        tries = 0
        while True:
            if init_mode == "around_best":
                cand = draw_around()
            elif init_mode == "broad":
                cand = draw_broad()
            elif init_mode == "hybrid":
                cand = draw_around() if (k < nwalkers // 2) else draw_broad()
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
                     tstar_dummy=6000.0,
                     nwalkers=32, nsteps=4000, burn_in=1000, y_mode="Flam",
                     use_weights=True, n_cores=4, mp_prefer="fork",
                     random_seed=42, posterior_mode="data", mix_weight=0.3,
                     tdust_sigma_frac=0.5, log10_tau_sigma=0.5, log10_a_sigma=1.0,
                     init_mode="hybrid", init_scales=None, progress_every=None,
                     cache_dir=None, cache_ndigits=4, cache_max=5000, use_tmp=True,
                     run_tag=None):
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

    if cache_dir is None:
        cache_dir = str(Path(workdir) / "dusty_npz_cache")

    template = None
    template_mode = (template_path is not None)
    if template_mode:
        template = NugentIIPSeries(str(Path(template_path).resolve()))
        ndim = 3
    else:
        ndim = 4

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
    
    # best-fit from grid
    grid_df = grid_df.sort_values("chi2_red").reset_index(drop=True)
    best_row = grid_df.iloc[0]

    if template_mode:
        best = {
            "tdust": float(best_row["tdust"]),
            "log10_tau": np.log10(float(best_row["tau"])),
            "log10_a": np.log10(float(best_row["scale"])),
        }
        prior_config = dict(
            tdust_bounds=(50., 1500.),
            log10_tau_bounds=(-4., 2.),
            log10_a_bounds=(-30., 30.),
            best=best,
            prior_mode=posterior_mode,
            mix_weight=mix_weight,
            tdust_sigma_frac=tdust_sigma_frac,
            log10_tau_sigma=log10_tau_sigma,
            log10_a_sigma=log10_a_sigma,
        )
    else:
        best = {
            "tstar": float(best_row["tstar"]),
            "tdust": float(best_row["tdust"]),
            "log10_tau": np.log10(float(best_row["tau"])),
            "log10_a": np.log10(float(best_row["scale"])),
        }
        prior_config = dict(
            tstar_bounds=(2000., 12000.),
            tdust_bounds=(100., 1500.),
            log10_tau_bounds=(-4., 2.),
            log10_a_bounds=(-30., 30.),
            best=best,
            prior_mode=posterior_mode,
            mix_weight=mix_weight,
            tstar_sigma_frac=0.3,
            tdust_sigma_frac=tdust_sigma_frac,
            log10_tau_sigma=log10_tau_sigma,
            log10_a_sigma=log10_a_sigma,
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

    if burn_in >= nsteps:
        raise ValueError("burn_in must be less than nsteps.")
    
    ctx = _pick_mp_context(prefer=mp_prefer)
    ncores = min(int(n_cores), nwalkers)

    with ctx.Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(sed, dusty_runner, prior_config, y_mode, use_weights,
                  template, template_tag, tstar_dummy, shell_thickness),
            pool=pool
        )
        print(f"Running MCMC: ndim={ndim}, nwalkers={nwalkers}, nsteps={nsteps}...", flush=True)

        if progress_every is not None:
            state = p0
            for i, state in enumerate(sampler.sample(state, iterations=nsteps), start=1):
                if (i == 1) or (i % progress_every == 0) or (i == nsteps):
                    print(f"[emcee] step {i}/{nsteps}", flush=True)
        else:
            sampler.run_mcmc(p0, nsteps, progress=True)

    # sampler = emcee.EnsembleSampler(nwalkers,
    #                                 ndim,
    #                                 log_probability,
    #                                 args=(sed, dusty_runner, prior_config, y_mode))
    # print(f"Running MCMC: nwalkers={nwalkers}, nsteps={nsteps}...")
    # sampler.run_mcmc(p0, nsteps, progress=True)

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
    if ndim == 3:
        tdust_samples = samples[:, 0]
        log10_tau_samples = samples[:, 1]
        log10_a_samples = samples[:, 2]
        tstar_samples = np.full_like(tdust_samples, float(tstar_dummy))
    else:
        tstar_samples = samples[:, 0]
        tdust_samples = samples[:, 1]
        log10_tau_samples = samples[:, 2]
        log10_a_samples = samples[:, 3]

    tau_samples = 10.0 ** log10_tau_samples
    a_samples = 10.0 ** log10_a_samples

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
    )

    return results