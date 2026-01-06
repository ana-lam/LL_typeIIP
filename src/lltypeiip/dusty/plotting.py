import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from astropy import constants as const

from ..sed.build import _prepare_sed_xy
from ..sed.plotting import plot_sed
from .scaling import compute_chi2


def plot_best_fit_dusty_model(sed, df, y_mode="Flam", top_n=3, keep_sed_limits=False,
                              y_padding_frac=0.5, logx=True, logy=True, secax=False,
                              savepath=None, mcmc_results=None, dusty_runner=None,
                              mcmc_sample_mode="map"):
    """
    Plot the best N models (after scaling) over the SED and optionally overlay
    best fit from MCMC results.
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

        # for nice formatting

        exp = int(np.floor(np.log10(abs(m.scale))))
        a0 = m.scale / 10**exp

        label = (
                    r"Grid: "
                    rf"$T*: {m.Tstar}\,\mathrm{{K}},\ "
                    rf"T_\mathrm{{dust}}: {m.Tdust}\,\mathrm{{K}},\ "
                    rf"\tau: {m.tau},$"
                    "\n"
                    rf"$a: {a0:.1f}\times10^{{{exp}}} \ $ "
                    rf"$\chi^2 = {r['chi2']:.1f},\ \chi^2_\mathrm{{red}} = {r['chi2_red']:.2f}$"
                )
        ax.plot(m.x_plot, m.y_scaled, lw=2, color=palette[i % len(palette)], linestyle=linestyle, alpha=alpha,
                label=label)

    # MCMC best fit overlay
    y_mcmc_scaled = None
    x_mcmc_plot = None
    if mcmc_results is not None and dusty_runner is not None:
        samples = mcmc_results['samples']
        logp = mcmc_results['log_prob']

        if mcmc_sample_mode == "median":
            # medians across posterior
            tstar_m = np.median(samples[:, 0])
            tdust_m = np.median(samples[:, 1])
            log10_tau_m = np.median(samples[:, 2])
            log10_a_m = np.median(samples[:, 3])
        else:
            # maximum a posteriori
            max_idx = np.argmax(logp)
            tstar_m = samples[max_idx, 0]
            tdust_m = samples[max_idx, 1]
            log10_tau_m = samples[max_idx, 2]
            log10_a_m = samples[max_idx, 3]
        
        tau_m = 10.0**log10_tau_m
        a_m = 10.0**log10_a_m
        lam_um_m, lamFlam_m, r1_m = dusty_runner.evaluate_model(tstar_m, tdust_m, tau_m)
        
        if lam_um_m is not None:
            # get analytic scale and chi^2 in the same way as grid models
            scale_m, chi2_m = compute_chi2(
                lam_um_m, lamFlam_m, sed, a=a_m, y_mode=y_mode, use_weights=True
            )

            if np.isfinite(chi2_m):
                if y_mode == "Flam":
                    x_mcmc_plot = lam_um_m
                    y_mcmc_scaled = scale_m * lamFlam_m
                elif y_mode == "Fnu":
                    lam_cm_m = lam_um_m * 1e-4
                    nu_m = const.c.cgs.value / lam_cm_m
                    Fnu_m = lamFlam_m * (lam_cm_m / const.c.cgs.value)

                    order = np.argsort(nu_m)
                    nu_m = nu_m[order]
                    Fnu_m = Fnu_m[order]

                    x_mcmc_plot = nu_m
                    y_mcmc_scaled = scale_m * Fnu_m

                def q16_50_84(x):
                    q16, q50, q84 = np.percentile(x, [16, 50, 84])
                    return q50, q50 - q16, q84 - q50   # median, -err, +err

                T_med,  T_m_err,  T_p_err  = q16_50_84(mcmc_results["tstar"])
                Td_med, Td_m_err, Td_p_err = q16_50_84(mcmc_results["tdust"])
                tau_med, tau_m_err, tau_p_err = q16_50_84(mcmc_results["tau"])
                a_med, a_m_err, a_p_err = q16_50_84(mcmc_results["a"])

                # for nice formatting
                exp = int(np.floor(np.log10(abs(a_med))))
                a0 = a_med / 10**exp
                a0_m = a_m_err / 10**exp
                a0_p = a_p_err / 10**exp

                N_data = len(sed["lam"])  # or however many SED points you actually use
                chi2_red_m = chi2_m / max(N_data - 1, 1)

                label = (
                    r"MCMC: "
                    rf"$T*: {T_med:.0f}_{{-{T_m_err:.0f}}}^{{+{T_p_err:.0f}}}\,\mathrm{{K}},\ "
                    rf"T_\mathrm{{dust}}: {Td_med:.0f}_{{-{Td_m_err:.0f}}}^{{+{Td_p_err:.0f}}}\,\mathrm{{K}},\ "
                    rf"\tau: {tau_med:.2f}_{{-{tau_m_err:.2f}}}^{{+{tau_p_err:.2f}}},$"
                    "\n"
                    rf"$a = {a0:.1f}_{{-{a0_m:.1f}}}^{{+{a0_p:.1f}}}\times10^{{{exp}}} \ $"
                    rf"$\chi^2 = {chi2_m:.1f},\ \chi^2_\mathrm{{red}} = {chi2_red_m:.2f}$"
                )
                
                ax.plot(x_mcmc_plot, y_mcmc_scaled, 
                        lw=2, color="deeppink", linestyle='-', alpha=1,
                        label=label)
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