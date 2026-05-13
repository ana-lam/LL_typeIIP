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
    if not np.isfinite(lnN):
        # Gaussian is -inf here; mixture falls back to uniform component
        return np.log(1.0 - w)
    return np.logaddexp(np.log(w) + lnN, np.log(1.0 - w))

def log_prior(theta, tstar_bounds=(1000., 10000.), #1000, 10000.
              tdust_bounds=(100., 1500.),
              log10_tau_bounds=(-4, 2.),
              log10_a_bounds=(-20., 0.),
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

    theta = np.asarray(theta, float)

    if theta.size == 4:
        tstar, tdust, log10_tau, log10_a = theta
        use_tstar = True
    # do not sample Tstar for Spectrum=5
    elif theta.size == 3:
        tdust, log10_tau, log10_a = theta
        tstar = None
        use_tstar = False
    else:
        return -np.inf
    
    # hard bounds
    if use_tstar:
        if not np.isfinite(_log_uniform_within(tstar, tstar_bounds)):
            return -np.inf
    if not np.isfinite(_log_uniform_within(tdust, tdust_bounds)):
        return -np.inf
    if not np.isfinite(_log_uniform_within(log10_tau, log10_tau_bounds)):
        return -np.inf
    if not np.isfinite(_log_uniform_within(log10_a, log10_a_bounds)):
        return -np.inf

    if prior_mode == "data":
        return 0.0
    
    # best for anchored and mixture priors
    if best is None:
        raise ValueError("best parameter values must be provided for anchored or mixture priors")
    
    tstar_sigma = (tstar_sigma_frac * best["tstar"]) if (use_tstar and tstar_sigma_frac) else None
    tdust_sigma = (tdust_sigma_frac * best["tdust"]) if tdust_sigma_frac else None

    ln_tstar = _log_normal(tstar, best.get("tstar", 0.0), tstar_sigma) if use_tstar else 0.0
    ln_tdust = _log_normal(tdust, best['tdust'], tdust_sigma)
    ln_log10_tau = _log_normal(log10_tau, best["log10_tau"], log10_tau_sigma)
    ln_log10_a = _log_normal(log10_a, best["log10_a"], log10_a_sigma)

    if prior_mode == "anchored":
        return ln_tstar + ln_tdust + ln_log10_tau + ln_log10_a
    
    if prior_mode == "mixture":
        w = float(mix_weight)
        if not (0.0 < w < 1.0):
            raise ValueError("mix_weight must be between 0 and 1")

        lp = 0.0
        if use_tstar:
            lp += _log_mix(ln_tstar, w)
        lp += _log_mix(ln_tdust, w)
        lp += _log_mix(ln_log10_tau, w)
        lp += _log_mix(ln_log10_a, w)
        return lp 

    raise ValueError("prior_mode must be one of: 'data', 'anchored', 'mixture'.")