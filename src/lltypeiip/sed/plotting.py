import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..config import SED_COLORS, SED_MARKERS
from ..photometry.plotting import plot_combined_lc
from .build import _prepare_sed_xy


def plot_sed(sed, ax=None, y_mode ="Fnu", logy=False, logx=False,
    title_prefix="SED", secax=False, savepath=None, legend_outside=True,
    inset_lc=None, inset_options=None):
    """
    y_mode='Fnu'  -> Fnu vs nu (mJy, Hz)
    y_mode='Flam' -> Flam vs λ (cgs/Ang, Ang)

    inset_lc : dict, optional
        Provide light-curve data to embed as an inset. Expected keys:
        {"ztf_resdict": ..., "wise_resdict": ..., "oid": ...}.
        When given, an inset uses plot_combined_lc(mode="overlay") and
        draws a red vertical line at sed["mjd"].
    inset_options : dict, optional
        Extra options forwarded to plot_combined_lc (e.g., logy, scale_wise).
    """

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        created_ax = True

    x, y, ey, x_label, y_label = _prepare_sed_xy(sed, y_mode=y_mode)
    bands = np.array(sed["bands"])
    is_ul = np.array(sed["is_ul"])
    dt = np.array(sed["dt_labels"])

    # detections per band
    for b in np.unique(bands):
        sel = (bands == b) & (~is_ul)
        if np.any(sel):
            ln = ax.errorbar(x[sel], y[sel], yerr=ey[sel],
                             fmt=SED_MARKERS.get(b, "o"),
                             color=SED_COLORS.get(b, "black"),
                             mec=SED_COLORS.get(b, "black"),
                             mfc=SED_COLORS.get(b, "black"),
                             linestyle="none", label=b + f" ({dt[sel][0]})")

    # upper limits
    for b in np.unique(bands):
        sel = (bands == b) & (is_ul)
        if np.any(sel):
            ln = ax.errorbar(x[sel], y[sel], yerr=None, uplims=True,
                             fmt="v", markersize=7,
                             color=SED_COLORS.get(b, "black"),
                             mec=SED_COLORS.get(b, "black"),
                             mfc=(0,0,0,0), linestyle="none", label=f"{b} upper limit")
    if secax:  
        # secondary axis      
        if y_mode == "Fnu":
            secax = ax.secondary_xaxis(
                'top',
                functions=(lambda nu: (const.c.value/nu)*1e6,        # ν [Hz] -> λ [µm]
                        lambda lam_um: const.c.value/(lam_um*1e-6)) # λ [µm] -> ν [Hz]
            )
            secax.set_xlabel(r"$\lambda\ (\mu\mathrm{m})$")
        elif y_mode == "Flam":
            secax = ax.secondary_xaxis(
                'top',
                functions=(lambda lam_um: const.c.value/(lam_um*1e-6),  # λ [µm] -> ν [Hz]
                        lambda nu: (const.c.value/nu)*1e6)           # ν [Hz] -> λ [µm]
            )
            secax.set_xlabel(r"$\nu\ (\mathrm{Hz})$")


    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f"{sed['oid']}: {title_prefix} at MJD {sed['mjd']:.2f}", fontsize=13)
    ax.grid(True, alpha=0.4)

    for line in ax.lines:
        mfc = line.get_markerfacecolor()
        mec = line.get_markeredgecolor()
        if mfc is None or mfc == "none":
            continue
        if isinstance(mfc, (tuple, list)) and len(mfc) == 4:
            r, g, b, _ = mfc
        elif isinstance(mfc, str):
            import matplotlib.colors as mcolors
            r, g, b, _ = mcolors.to_rgba(mfc)
        else:
            continue
        line.set_markerfacecolor((r, g, b, 0.3))   # semi-transparent fill
        line.set_markeredgecolor((r, g, b, 1.0))   # solid outline
        line.set_markeredgewidth(1.2)
        line.set_markersize(7)

    # legend: unique entries
    handles, lbls = ax.get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(handles, lbls):
        if l not in seen:
            H.append(h)
            L.append(l)
            seen.add(l)
    if H:
        if legend_outside:
                ax.legend(
                    H,
                    L,
                    fontsize=9,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.2),
                    borderaxespad=0.0,
                    ncol=4,
                )
        else:
            ax.legend(H, L, fontsize=9)

    # optional inset LC
    if inset_lc is not None and isinstance(inset_lc, dict):
        ztf_resdict = inset_lc.get("ztf_resdict")
        wise_resdict = inset_lc.get("wise_resdict", {})
        if ztf_resdict is not None:
            inset_kwargs = {
                "ztf_flux": True,
                "mode": "overlay",
                "scale_wise": True,
                "baseline_ref": "wise_parity",
                "ref_band": "r",
                "logy": True,
                "labels": False,
                "mark_plateau_end": True,
                "mark_tail_start": True,
            }
            if inset_options:
                inset_kwargs.update(inset_options)
                print(inset_kwargs)

            ax_inset = inset_axes(ax, width="30%", height="30%", loc="upper right", borderpad=1.1)
            plot_combined_lc(
                ztf_resdict,
                wise_resdict,
                oid=inset_lc.get("oid", sed.get("oid", "source")),
                ax=ax_inset,
                **inset_kwargs,
            )
            ax_inset.axvline(sed["mjd"], color="red", lw=1.75, alpha=0.75)
            ax_inset.grid(True, alpha=0.3)

    plt.tight_layout()

    if savepath:
            plt.savefig(savepath, bbox_inches="tight")
            print(f"Saved plot to {savepath}")
    
    if created_ax:
        plt.show()

    return ax