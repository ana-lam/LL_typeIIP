

import numpy as np


def log_prior(theta, tstar_bounds=(2000., 12000.),
                      tdust_bounds=(100., 1500.),
                      log10_tau_bounds=(-4., 2.),
                      log10_a_bounds=(-30., 30.),
                      best=None, gauss_frac=None,
                      a_prior_sigma_dex=None):
    """
    theta = (tstar, tdust, log10_tau)
    If best and gauss_frac are provided, use a Gaussian prior centered on best
    """
    if len(theta) != 4:
        tstar, tdust, log10_tau = theta
        log10_a = None
    else:
        tstar, tdust, log10_tau, log10_a = theta
    
    if not (tstar_bounds[0] <= tstar <= tstar_bounds[1]):
        return -np.inf
    if not (tdust_bounds[0] <= tdust <= tdust_bounds[1]):
        return -np.inf
    if not (log10_tau_bounds[0] <= log10_tau <= log10_tau_bounds[1]):
        return -np.inf
    
    if log10_a is not None:
        if not (log10_a_bounds[0] <= log10_a <= log10_a_bounds[1]):
            return -np.inf
    
    lp = 0.0

    if best is not None and gauss_frac is not None:
        sig_tstar = gauss_frac.get("tstar", 0.0) * best["tstar"]
        sig_tdust = gauss_frac.get("tdust", 0.0) * best["tdust"]
        sig_log10_tau = gauss_frac.get("log10_tau", 0.0)

        if log10_a is not None:
            sig_log10_a = gauss_frac.get("log10_a", 0.0) 

        if sig_tstar > 0:
            lp += -0.5 * ((tstar - best["tstar"])/sig_tstar)**2
        if sig_tdust > 0:
            lp += -0.5 * ((tdust - best["tdust"])/sig_tdust)**2
        if sig_log10_tau > 0:
            lp += -0.5 * ((log10_tau - best["log10_tau"])/sig_log10_tau)**2
    
    if best is not None and (log10_a is not None) and (a_prior_sigma_dex is not None):
        lp += -0.5 * ((log10_a - best["log10_a"])/a_prior_sigma_dex)**2

    return lp