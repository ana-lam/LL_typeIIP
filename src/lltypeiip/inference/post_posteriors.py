import numpy as np

from ..physics.bolometric import integrate_Fbol_from_lamFlam, Fbol_to_Lbol
from ..physics.nickel_mass import MNi_from_tail


def posterior_Lbol_MNi(mcmc_results, dusty_runner, z,
                       t_days, max_draws=500, seed=42):
    """
    Convert MCMC posterior samples into posterior samples of F_bol, L_bol, and M_Ni.
    """

    rng = np.random.default_rng(seed)
    samples = mcmc_results["samples"]
    logp = mcmc_results["log_prob"]

    n = samples.shape[0]
    if n == 0:
        raise ValueError("No samples in mcmc_results.")

    if (max_draws is not None) and (n > max_draws):
        idx = rng.choice(n, size=max_draws, replace=False)
        samples = samples[idx]
        logp = logp[idx]

    Fbol = np.full(len(samples), np.nan, float)
    Lbol = np.full(len(samples), np.nan, float)
    MNi = np.full(len(samples), np.nan, float)

    for i, th in enumerate(samples):
        tstar, tdust, tau, a = th
        lam_um, lamFlam, r1 = dusty_runner.evaluate_model(tstar, tdust, tau)

        if lam_um is None:
            continue

        # scale
        lamFlam_scaled = a * lamFlam

        # integrate to get F_bol
        F_bol = integrate_Fbol_from_lamFlam(lam_um, lamFlam_scaled)
        L_bol = Fbol_to_Lbol(F_bol, z)
        M_Ni = MNi_from_tail(L_bol, t_days)

        Fbol[i] = F_bol
        Lbol[i] = L_bol
        MNi[i] = M_Ni

    return dict(
        Fbol=Fbol,
        Lbol=Lbol,
        MNi=MNi,
        log_prob=logp
    )