import os
from multiprocessing import get_context

import emcee
import numpy as np
from astropy import constants as const

from ..dusty import DustyRunner, compute_chi2
from ..sed.build import _prepare_sed_xy
from .priors import log_prior


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


def log_likelihood(theta, sed, dusty_runner,
                   y_mode="Flam", use_weights=True):
    """
    theta = (tstar, tdust, log10_tau, log10_a)
    """
    if len(theta) != 4:
        tstar, tdust, log10_tau = theta
        tau = 10.0**log10_tau
        a = None
    else:
        tstar, tdust, log10_tau, log10_a = theta
        tau = 10.0**log10_tau
        a = 10.0**log10_a

    lam_um, lamFlam, r1 = dusty_runner.evaluate_model(tstar, tdust, tau)

    if lam_um is None:
        return -np.inf

    scale, chi2 = _ls_scale_and_chi2(
        lam_um, lamFlam, sed, a=a, y_mode=y_mode, use_weights=use_weights
    )
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2


def log_probability(theta, sed, dusty_runner, prior_config, y_mode="Flam"):
    lp = log_prior(theta, **prior_config)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, sed, dusty_runner, y_mode=y_mode)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_mcmc_for_sed(sed, grid_df, dusty_file_dir, workdir, 
                     dust_type="silicate", shell_thickness=2.0,
                     nwalkers=20, nsteps=2000, burn_in=500, y_mode="Flam",
                     n_cores=None, random_seed=42):
    """
    Run emcee for a given SED using DUSTY models.
    """

    np.random.seed(random_seed)

    dusty_runner = DustyRunner(base_workdir=workdir,
                               dusty_file_dir=dusty_file_dir,
                               dust_type=dust_type,
                               shell_thickness=shell_thickness)
    
    # best-fit from grid
    best_row = grid_df.iloc[0]
    best = {
        "tstar": float(best_row["tstar"]),
        "tdust": float(best_row["tdust"]),
        "log10_tau": np.log10(float(best_row["tau"])),
        "log10_a": np.log10(float(best_row["scale"]))
    }

    # prior configuration
    prior_config = dict(
        tstar_bounds=(2000., 12000.),
        tdust_bounds=(100., 1500.),
        log10_tau_bounds=(-4., 2.),
        best=best,
        gauss_frac={"tstar": 0.3, "tdust": 0.5, "log10_tau": 0.5}, # 30%, 50%, 50%
        a_prior_sigma_dex=1.0
    )

    # initial positions from best
    ndim = 4 # tstar, tdust, log10_tau, log10_a
    p0 = np.zeros((nwalkers, ndim))
    p0[:, 0] = best["tstar"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    p0[:, 1] = best["tdust"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    p0[:, 2] = best["log10_tau"] + 0.2 * np.random.randn(nwalkers)
    p0[:, 3] = best["log10_a"] + 0.2 * np.random.randn(nwalkers)
    
    for k in range(nwalkers):
        lp = log_prior(p0[k], **prior_config)
        if not np.isfinite(lp):
            p0[k, 0] = np.random.uniform(*prior_config["tstar_bounds"])
            p0[k, 1] = np.random.uniform(*prior_config["tdust_bounds"])
            p0[k, 2] = np.random.uniform(*prior_config["log10_tau_bounds"])
    

    if n_cores is None:
        ncores = min(os.cpu_count() or 1, nwalkers) # don't exceed specified nwalkers
    else:
        ncores = n_cores
    
    ctx = get_context("fork")

    with ctx.Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(sed, dusty_runner, prior_config, y_mode),
            pool=pool
        )
        sampler.run_mcmc(p0, nsteps, progress=True)

    # sampler = emcee.EnsembleSampler(nwalkers,
    #                                 ndim,
    #                                 log_probability,
    #                                 args=(sed, dusty_runner, prior_config, y_mode))
    # print(f"Running MCMC: nwalkers={nwalkers}, nsteps={nsteps}...")
    # sampler.run_mcmc(p0, nsteps, progress=True)

    # discard burn-in and flatten
    samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burn_in, thin=1, flat=True)

    # transform back to (tstar, tdust, tau)
    tstar_samples = samples[:, 0]
    tdust_samples = samples[:, 1]
    log10_tau_samples = samples[:, 2]
    log10_a_samples = samples[:, 3]
    tau_samples = 10.0**log10_tau_samples
    a_samples = 10.0**log10_a_samples

    results = dict(
        samples=samples,
        log_prob=log_prob_samples,
        tstar=tstar_samples,
        tdust=tdust_samples,
        tau=tau_samples,
        a=a_samples,
        grid_best=best,
        sampler=sampler
    )

    return results