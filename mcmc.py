import numpy as np
import emcee
from pydusty.dusty import Dusty
from SED_functions import _prepare_sed_xy 
from dusty.dusty_runner import DustyRunner

def compute_scale_and_chi2(lam_um, lamFlam, sed, y_mode="Flam", use_weights=True):
    """
    Analytic best-fit scale + chi^2.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)

    if y_mode == "Flam":
        x_mod = lam_um
        y_mod = lamFlam
    elif y_mode == "Fnu":
        from astropy import constants as const
        lam_cm = lam_um * 1e-4
        nu_mod = const.c.cgs.value / lam_cm
        Fnu_mod = lamFlam * (lam_cm / const.c.cgs.value)
        order = np.argsort(nu_mod)
        x_mod = nu_mod[order]
        y_mod = Fnu_mod[order]

    # overlap in x
    mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())
    if not np.any(mask_data):
        return np.nan, np.inf

    x_data = x_sed[mask_data]
    y_data = y_sed[mask_data]
    ey_data = ey_sed[mask_data]

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
    if dof > 0:
        chi2_red = chi2 / dof
    else:
        chi2_red = np.inf
    
    return float(a), float(chi2)

def log_likelihood(theta, sed, dusty_runner,
                   y_mode="Flam", use_weights=True):
    """
    theta = (tstar, tdust, log10_tau)
    """
    tstar, tdust, log10_tau = theta
    tau = 10.0**log10_tau

    lam_um, lamFlam, r1 = dusty_runner.evaluate_model(tstar, tdust, tau)

    if lam_um is None:
        return -np.inf

    scale, chi2 = compute_scale_and_chi2(lam_um, lamFlam, sed,
                                         y_mode=y_mode,
                                         use_weights=use_weights)
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2

def log_prior(theta, tstar_bounds=(2000., 12000.),
                      tdust_bounds=(100., 1500.),
                      log10_tau_bounds=(-4., 2.),
                      best=None, gauss_frac=None):
    """
    theta = (tstar, tdust, log10_tau)
    If best and gauss_frac are provided, use a Gaussian prior centered on best
    """
    tstar, tdust, log10_tau = theta
    
    if not (tstar_bounds[0] <= tstar <= tstar_bounds[1]):
        return -np.inf
    if not (tdust_bounds[0] <= tdust <= tdust_bounds[1]):
        return -np.inf
    if not (log10_tau_bounds[0] <= log10_tau <= log10_tau_bounds[1]):
        return -np.inf
    
    lp = 0.0

    if best is not None and gauss_frac is not None:
        sig_tstar = gauss_frac.get("tstar", 0.0) * best["tstar"]
        sig_tdust = gauss_frac.get("tdust", 0.0) * best["tdust"]
        sig_log10_tau = gauss_frac.get("log10_tau", 0.0)

        if sig_tstar > 0:
            lp += -0.5 * ((tstar - best["tstar"])/sig_tstar)**2
        if sig_tdust > 0:
            lp += -0.5 * ((tdust - best["tdust"])/sig_tdust)**2
        if sig_log10_tau > 0:
            lp += -0.5 * ((log10_tau - best["log10_tau"])/sig_log10_tau)**2

    return lp

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
                     nwalkers=32, nsteps=2000, burn_in=500, y_mode="Flam",
                     random_seed=42):
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
        "log10_tau": np.log10(float(best_row["tau"]))
    }

    # prior configuration
    prior_config = dict(
        tstar_bounds=(2000., 12000.),
        tdust_bounds=(100., 1500.),
        log10_tau_bounds=(-4., 2.),
        best=best,
        gauss_frac={"tstar": 0.3, "tdust": 0.5, "log10_tau": 0.5} # 30%, 50%, 50%
    )

    # initial positions from best
    ndim = 3 # tstar, tdust, log10_tau
    p0 = np.zeros((nwalkers, ndim))
    p0[:, 0] = best["tstar"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    p0[:, 1] = best["tdust"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    p0[:, 2] = best["log10_tau"] + 0.2 * np.random.randn(nwalkers)
    
    for k in range(nwalkers):
        lp = log_prior(p0[k], **prior_config)
        if not np.isfinite(lp):
            p0[k, 0] = np.random.uniform(*prior_config["tstar_bounds"])
            p0[k, 1] = np.random.uniform(*prior_config["tdust_bounds"])
            p0[k, 2] = np.random.uniform(*prior_config["log10_tau_bounds"])
    
    sampler = emcee.EnsembleSampler(nwalkers,
                                    ndim,
                                    log_probability,
                                    args=(sed, dusty_runner, prior_config, y_mode))
    print(f"Running MCMC: nwalkers={nwalkers}, nsteps={nsteps}...")
    sampler.run_mcmc(p0, nsteps, progress=True)

    # discard burn-in and flatten
    samples = sampler.get_chain(discard=burn_in, thin=1, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burn_in, thin=1, flat=True)

    # transform back to (tstar, tdust, tau)
    tstar_samples = samples[:, 0]
    tdust_samples = samples[:, 1]
    log10_tau_samples = samples[:, 2]
    tau_samples = 10.0**log10_tau_samples

    results = dict(
        samples=samples,
        log_prob=log_prob_samples,
        tstar=tstar_samples,
        tdust=tdust_samples,
        tau=tau_samples,
        grid_best=best
    )

    return results