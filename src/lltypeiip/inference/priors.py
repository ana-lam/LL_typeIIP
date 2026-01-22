import numpy as np

def _log_uniform_within(x, bounds):
    lo, hi = bounds
    return 0.0 if (lo <= x <= hi) else -np.inf

def _log_normal(x, mu, sigma):
    if sigma is None or sigma <= 0:
        return 0.0
    z = (x - mu) / sigma
    return -0.5 * z * z

def _log_mix(lnN, w):
    # Uniform part is constant 0 inside bounds (already ensured); use logaddexp for stability.
    return np.logaddexp(np.log(w) + lnN, np.log(1.0 - w))

def log_prior(theta, tstar_bounds=(2000., 15000.), #1000, 15000.
              tdust_bounds=(50., 1500.),
              log10_tau_bounds=(-4, 2.),
              log10_a_bounds=(-30., 30.),
              best=None, prior_mode="data", 
              tstar_sigma_frac=None, tdust_sigma_frac=None,
              log10_tau_sigma=None, log10_a_sigma=None,
              mix_weight=0.3):
    """
    theta = (tstar, tdust, log10_tau, log10_a)

    prior_mode:
        - "data": uniform within data bounds (data-driven prior)
        - "anchored": bounds + Gaussian penalties around 'best' grid fit (anchored prior)
        - "mixture": mixture of data-driven and anchored priors
            p = mix_weight * anchored + (1-mix_weight) * data-driven
    """

    if len(theta) != 4:
        print("Error: theta must have 4 elements (tstar, tdust, log10_tau, log10_a)")
        tstar, tdust, log10_tau = theta
        log10_a = None
    else:
        tstar, tdust, log10_tau, log10_a = theta

    lp = 0.0
    lp += _log_uniform_within(tstar, tstar_bounds)
    if not np.isfinite(lp):
        return -np.inf
    lp += _log_uniform_within(tdust, tdust_bounds)
    if not np.isfinite(lp):
        return -np.inf
    lp += _log_uniform_within(log10_tau, log10_tau_bounds)
    if not np.isfinite(lp):
        return -np.inf
    if log10_a is not None:
        lp += _log_uniform_within(log10_a, log10_a_bounds)
        if not np.isfinite(lp):
            return -np.inf

    if prior_mode == "data":
        return 0.0

    if best is None:
        raise ValueError("best parameter values must be provided for anchored or mixture priors")
    
    # resolve sigmas
    tstar_sigma = (tstar_sigma_frac * best["tstar"]) if (tstar_sigma_frac is not None) else None
    tdust_sigma = (tdust_sigma_frac * best["tdust"]) if (tdust_sigma_frac is not None) else None

    ln_tstar = _log_normal(tstar, best["tstar"], tstar_sigma)
    ln_tdust = _log_normal(tdust, best["tdust"], tdust_sigma)
    ln_log10_tau = _log_normal(log10_tau, best["log10_tau"], log10_tau_sigma)
    ln_log10_a = _log_normal(log10_a, best["log10_a"], log10_a_sigma)

    if prior_mode == "anchored":
        lp += ln_tstar + ln_tdust + ln_log10_tau + ln_log10_a
        return lp

    if prior_mode == "mixture":
        w = float(mix_weight)
        if not (0.0 < w < 1.0):
            raise ValueError("mix_weight must be between 0 and 1")

        lp += _log_mix(ln_tstar, w) + _log_mix(ln_tdust, w) + \
                _log_mix(ln_log10_tau, w) + _log_mix(ln_log10_a, w)
        return lp
    
    raise ValueError("prior_mode must be one of: 'data', 'anchored', 'mixture'.")


# def log_prior(theta, tstar_bounds=(2000., 12000.),
#                       tdust_bounds=(100., 1500.),
#                       log10_tau_bounds=(-4., 2.),
#                       log10_a_bounds=(-30., 30.),
#                       best=None, gauss_frac=None,
#                       a_prior_sigma_dex=None):
#     """
#     theta = (tstar, tdust, log10_tau)
#     If best and gauss_frac are provided, use a Gaussian prior centered on best
#     """
#     if len(theta) != 4:
#         tstar, tdust, log10_tau = theta
#         log10_a = None
#     else:
#         tstar, tdust, log10_tau, log10_a = theta
    
#     if not (tstar_bounds[0] <= tstar <= tstar_bounds[1]):
#         return -np.inf
#     if not (tdust_bounds[0] <= tdust <= tdust_bounds[1]):
#         return -np.inf
#     if not (log10_tau_bounds[0] <= log10_tau <= log10_tau_bounds[1]):
#         return -np.inf
    
#     if log10_a is not None:
#         if not (log10_a_bounds[0] <= log10_a <= log10_a_bounds[1]):
#             return -np.inf
    
#     lp = 0.0

#     if best is not None and gauss_frac is not None:
#         sig_tstar = gauss_frac.get("tstar", 0.0) * best["tstar"]
#         sig_tdust = gauss_frac.get("tdust", 0.0) * best["tdust"]
#         sig_log10_tau = gauss_frac.get("log10_tau", 0.0)

#         if log10_a is not None:
#             sig_log10_a = gauss_frac.get("log10_a", 0.0) 

#         if sig_tstar > 0:
#             lp += -0.5 * ((tstar - best["tstar"])/sig_tstar)**2
#         if sig_tdust > 0:
#             lp += -0.5 * ((tdust - best["tdust"])/sig_tdust)**2
#         if sig_log10_tau > 0:
#             lp += -0.5 * ((log10_tau - best["log10_tau"])/sig_log10_tau)**2
    
#     if best is not None and (log10_a is not None) and (a_prior_sigma_dex is not None):
#         lp += -0.5 * ((log10_a - best["log10_a"])/a_prior_sigma_dex)**2

#     return lp