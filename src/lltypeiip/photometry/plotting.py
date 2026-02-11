import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
from astropy.stats import sigma_clipped_stats
from pathlib import Path

from ..config import config, SED_COLORS, SED_MARKERS
from .extinction import calculate_distance_modulus
from .wise import subtract_wise_parity_baseline
from .ztf import convert_ZTF_mag_mJy

##########################################################################
## -------------- ZTF PHOTOMETRY PLOTTING FUNCTIONS ------------------- ##
##########################################################################


## Using a lot of Viraj's code from LSST CCA Summer School

FILTER_COLORS = {1: "green", 2: "red", 3: "orange"}
FILTER_LABELS = {1: 'g', 2: 'r', 3: 'i'}
FILTER_MARKERS = {1: 'o', 2: 'X', 3: 'D'}
FILTER_SIZES = {1: 30, 2: 60, 3: 90}


def plot_forced_lc(resdict, oid="ZTF source", xlim=(None, None), ax=None, show=True, flux=False,
                   ylim=(None, None), SNU=5.0):
    """
    Plot ZTF forced photometry light curves (apparent magnitude or flux).

    Parameters
    ----------
    resdict : dict
        Dictionary containing forced photometry data by filter.
        Expected structure:
            {
              "ZTF_g": {"mjd": [...], "mag": [...], "mag_err": [...],
                        "mag_ul": [...], "flux": [...], "flux_err": [...],
                        "flux_mJy": [...], "flux_err_mJy": [...],
                        "lim_flux_mJy": [...]},
              "ZTF_r": {...},
              ...
            }
    oid : str, optional
        Object ID or name to show in plot title.
    xlim : tuple, optional
        (xmin, xmax) for x-axis (MJD). Default (None, None) = auto.
    ax : matplotlib.axes.Axes, optional
        Existing matplotlib axis. If None, a new figure is created.
    show : bool, optional
        Whether to display the plot. If False and ax is provided, returns the axis.
    flux : bool, optional
        If True, plots in flux (mJy) instead of magnitudes.
    ylim : tuple, optional
        (ymin, ymax) for y-axis. Default (None, None) = auto.
    SNU : float, optional
        Signal-to-noise multiplier for computing flux limits
        when explicit limit fields are missing (default 5).

    Notes
    -----
    - Magnitude plots invert the y-axis (bright = up).
    - Handles both detections and non-detections (upper limits).
    - If `flux=True`, tries to use mJy fields first, otherwise raw DN units.
    """

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))
        created_ax = True

    # Use colors and markers from config
    colors = SED_COLORS
    markers = SED_MARKERS

    all_mjd = []
    all_y = []

    # avoid duplicate legend entries for limits per filter
    shown_limit_label = set()

    # if flux=True, check if mJy fields are available
    use_mJy = flux and any(("flux_mJy" in d) for d in resdict.values())

    for filt, data in resdict.items():
        color = colors.get(filt,"black")

        # --- Detections ---
        mask_det = ~np.isnan(data["mag"])
        if np.any(mask_det):
            if flux:
                if "flux_mJy" in data and "flux_err_mJy" in data:
                    yvals = data["flux_mJy"][mask_det]
                    yerrs = data["flux_err_mJy"][mask_det]
                else:
                    yvals = data["flux"][mask_det]       # DN fallback
                    yerrs = data["flux_err"][mask_det]   # DN fallback
            else:
                yvals = data["mag"][mask_det]
                yerrs = data["mag_err"][mask_det]

            ax.errorbar(
                data["mjd"][mask_det],
                yvals,
                yerr=yerrs,
                color=color, label=filt,
                marker = markers.get(filt, "o"),
                linestyle='none'
            )
            all_mjd.extend(data["mjd"][mask_det])
            all_y.extend(yvals)

        # --- Non-detections (limits) ---
        # Use mag_ul when plotting mags; fall back to limiting_mag if mag_ul missing.
        # In flux space, prefer provided lim_flux_mJy; else compute snu*flux_err in same units.
        mask_nondet = np.isnan(data["mag"])

        if np.any(mask_nondet):
            if flux:
                if use_mJy and "lim_flux_mJy" in data:
                    yvals_lim = data["lim_flux_mJy"][mask_nondet]
                else:
                    yvals_lim = SNU * data['flux_err'][mask_nondet]
            else:
                if "mag_ul" in data:
                    yvals_lim = data["mag_ul"][mask_nondet]
                else:
                    yvals_lim = data.get("limiting_mag", np.full_like(data["mjd"], np.nan))[mask_nondet]

            # only plot finite limits
            finite = np.isfinite(yvals_lim)

            ax.scatter(
                data["mjd"][mask_nondet][finite],
                yvals_lim[finite],
                marker="v", alpha=0.5, color=color,
                label=f"lim. mag {filt}" if filt not in shown_limit_label else None,
                s=40
            )
            shown_limit_label.add(filt)
            all_mjd.extend(data["mjd"][mask_nondet][finite])
            all_y.extend(yvals_lim[finite])

    # --- Auto limits ---
    if all_mjd and all_y:
        min_mjd = xlim[0] if xlim[0] is not None else None
        max_mjd = xlim[1] if xlim[1] is not None else None

        if min_mjd is not None and max_mjd is not None:
            ax.set_xlim(min_mjd, max_mjd)

        if ylim[0] is not None and ylim[1] is not None:
            ax.set_ylim(ylim[0], ylim[1])
        else:
            # y-range: min/max mags + padding
            ymin = np.nanmin(all_y) - 2.0
            ymax = np.nanmax(all_y) + 1.0

            if flux:
                ax.set_ylim(ymin, ymax)
            else:   
                ax.set_ylim(ymax, ymin)  # flip so bright = up

    if created_ax:
        ax.set_title(f"ZTF Light Curve: {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        if flux:
            ax.set_ylabel("Flux (mJy)", fontsize=14)
        else:
            ax.set_ylabel("Apparent magnitude", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.4)
        
        if show:
            plt.show()

    if not created_ax:
        return ax

def plot_stamps(oid, lc_det, client):
    """
    Plot ZTF image stamps (science, reference, difference).
    
    Parameters
    ----------
    oid : str
        Object identifier.
    lc_det : pd.DataFrame
        Detection dataframe from ALeRCE with 'has_stamp' column.
    client : ALeRCE client
        ALeRCE API client instance.
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    # Find first detection with a valid stamp
    if "has_stamp" not in lc_det.columns or lc_det["has_stamp"].sum() == 0:
        print(f"No stamp available for {oid}")
        return

    try:
        candid = lc_det.loc[lc_det.has_stamp].sort_values("mjd").candid.iloc[0]
        stamps = client.get_stamps(oid, candid, format='HDUList')
        science, ref, difference = stamps[0].data, stamps[1].data, stamps[2].data
    except Exception as e:
        print(f"Failed to fetch stamps for {oid}: {e}")
        return

    # Plot the cutouts
    fig, ax = plt.subplots(ncols=3, figsize=(8, 4))
    titles = ["Science", "Reference", "Difference"]
    images = [science, ref, difference]

    for i in range(3):
        img = np.log1p(images[i])  # log scale with log1p for stability
        _, med, std = sigma_clipped_stats(img, sigma=3.0)
        ax[i].imshow(img, cmap='viridis', origin='lower')
        ax[i].set_title(titles[i])
        ax[i].axis("off")

    fig.suptitle(f"{oid}, candid: {candid}", fontsize=12, y=0.9)
    plt.tight_layout()
    plt.show()

def plot_lc(oid, SN_det, SN_nondet):
    """
    Plot ZTF alert light curve (detections and non-detections).
    
    Parameters
    ----------
    oid : str
        Object identifier.
    SN_det : pd.DataFrame
        Detection dataframe from ALeRCE with columns:
        mjd, fid, magpsf, sigmapsf, etc.
    SN_nondet : pd.DataFrame
        Non-detection dataframe from ALeRCE with columns:
        mjd, fid, diffmaglim.
    
    Returns
    -------
    None
        Displays matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Loop over whatever filters are actually present
    for fid in sorted(SN_det.fid.dropna().unique()):
        color = FILTER_COLORS.get(fid, "black")
        label = FILTER_LABELS.get(fid, f"fid={fid}")
        marker = FILTER_MARKERS.get(fid, "o")
        size = FILTER_SIZES.get(fid, 40)

        # --- Detections ---
        mask_det = (SN_det.fid == fid) & SN_det.magpsf.notna()
        if mask_det.any():
            ax.errorbar(
                SN_det.loc[mask_det, "mjd"],
                SN_det.loc[mask_det, "magpsf"],
                yerr=SN_det.loc[mask_det, "sigmapsf"],
                c=color, label=label,
                marker=marker, linestyle='none'
            )

        # --- Non-detections (limits) ---
        mask_nondet = (SN_nondet.fid == fid) & (SN_nondet.diffmaglim > 0)
        if mask_nondet.any():
            ax.scatter(
                SN_nondet.loc[mask_nondet, "mjd"],
                SN_nondet.loc[mask_nondet, "diffmaglim"],
                c=color, alpha=0.5, marker='v',
                label=f"lim.mag. {label}", s=size
            )

    ax.set_title(oid, fontsize=16)
    ax.set_xlabel("MJD", fontsize=14)
    ax.set_ylabel("Apparent magnitude", fontsize=14)

    # Flip y-axis so brighter = up
    ax.set_ylim(ax.get_ylim()[::-1])

    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.show()

def plot_intrinsic_forced_lc(resdict, oid="ZTF source", ax=None, show=True, show_comparison=False, 
                             flux=False, xlim=(None,None), ylim=(None, None), SNU=5.0):
    """
    Plot ZTF forced photometry intrinsic (absolute) light curve, after extinction- 
    and distance-corrections.

    Parameters
    ----------
    resdict : dict
        Result dictionary from intrinsic_lc_corrections().
        Expected structure:
            res["forced_corr"] = {
                "ZTF_g": {
                    "mjd": [...],
                    "mag": [...],
                    "mag_err": [...],
                    "mag_intrinsic": [...],
                    "A_total": [...],
                },
                "ZTF_r": {...},
                "ZTF_i": {...},
            }
    oid : str, optional
        Object name or ID for labeling.
    flux : bool, optional
        If True, plots fluxes (mJy) instead of magnitudes.
    xlim : tuple, optional
        (xmin, xmax) for x-axis. Default (None, None) = auto.
    ax : matplotlib.axes.Axes, optional
        Existing axis to plot into. If None, creates new figure.
    show : bool, optional
        Whether to display the plot interactively. Default True.
    ylim : tuple, optional
        (ymin, ymax) for y-axis. Default auto.
    show_comparison : bool, optional
        If True, overlays observed (apparent) magnitudes as hollow markers.

    Notes
    -----
    - Plots *absolute* magnitudes (distance- and extinction-corrected).
    - Inverts the y-axis (bright = up).
    """

    if not any("mag_intrinsic" in d for d in resdict.values()):
        print("No corrected light curve data found. Run intrinsic_lc_corrections() first.")
        return
    
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))
        created_ax = True

    if show_comparison and not flux:
        ax_abs = ax
        ax_app = ax.twinx()
    else:
        ax_abs = ax
        ax_app = None

    colors = {"ZTF_g":"green","ZTF_r":"red","ZTF_i":"orange"}
    markers = {"ZTF_g":"o","ZTF_r":"X","ZTF_i":"D"}

    all_mjd = []
    all_y = []

    # avoid duplicate legend entries for limits per filter
    shown_limit_label = set()

    # if flux=True, check if mJy fields are available
    use_mJy = flux and any(("flux_mJy" in d) for d in resdict.values())

    for filt, data in resdict.items():
        color = colors.get(filt,"black")
        marker = markers.get(filt, "o")

        mjd = data["mjd"]

        # --- Detections ---
        mask_det = ~np.isnan(data["mag"])
        if np.any(mask_det):
            if flux:
                if "flux_mJy_intrinsic" in data and "flux_err_mJy_intrinsic" in data:
                    yvals = data["flux_mJy_intrinsic"][mask_det]
                    yerrs = data["flux_err_mJy_intrinsic"][mask_det]
                    yvals_obs = data["flux_mJy"][mask_det]
                    yerrs_obs = data["flux_err_mJy"][mask_det]
                else:
                    yvals = data["flux_intrinsic"][mask_det]       # DN fallback
                    yerrs = data["flux_intrinsic_err"][mask_det]   # DN fallback
            else:
                yvals = data["mag_intrinsic"][mask_det]
                yerrs = data["mag_intrinsic_err"][mask_det]
                yvals_obs = data["mag"][mask_det]
                yerrs_obs = data["mag_err"][mask_det]

            # plot intrinsic detections
            ax_abs.errorbar(
                data["mjd"][mask_det],
                yvals,
                yerr=yerrs,
                color=color, label=filt,
                marker=marker,
                linestyle='none'
            )
            all_mjd.extend(data["mjd"][mask_det])
            all_y.extend(yvals)

            if show_comparison:
                # plot apparent mag too
                if show_comparison and ax_app is not None and not flux:
                    ax_app.errorbar(
                        mjd[mask_det], yvals_obs, yerr=yerrs_obs,
                        color=color, marker=marker,
                        linestyle='none', markerfacecolor='none',
                        alpha=0.6, label=f"{filt} (apparent)"
                    )

        # --- Non-detections (limits) ---
        # Use mag_ul when plotting mags; fall back to limiting_mag if mag_ul missing.
        # In flux space, prefer provided lim_flux_mJy; else compute snu*flux_err in same units.
        mask_nondet = np.isnan(data["mag_intrinsic"])

        if np.any(mask_nondet):
            if flux:
                if use_mJy and "lim_flux_intrinsic_mJy" in data:
                    yvals_lim = data["lim_flux_intrinsic_mJy"][mask_nondet]
                else:
                    yvals_lim = SNU * data['flux_intrinsic_err'][mask_nondet]
            else:
                if "mag_intrinsic_ul" in data:
                    yvals_lim = data["mag_intrinsic_ul"][mask_nondet]
                else:
                    yvals_lim = data.get("limiting_mag_intrinsic", np.full_like(data["mjd"], np.nan))[mask_nondet]

            # only plot finite limits
            finite = np.isfinite(yvals_lim)

            ax_abs.scatter(
                data["mjd"][mask_nondet][finite],
                yvals_lim[finite],
                marker="v", alpha=0.5, color=color,
                label=f"lim. mag {filt}" if filt not in shown_limit_label else None,
                s=40
            )
            shown_limit_label.add(filt)
            all_mjd.extend(data["mjd"][mask_nondet][finite])
            all_y.extend(yvals_lim[finite])

    # --- Auto limits ---
    if all_mjd and all_y:
        if xlim[0] is not None and xlim[1] is not None:
            ax_abs.set_xlim(xlim)
        if ylim[0] is not None and ylim[1] is not None:
            ax_abs.set_ylim(ylim)
        else:
            ymin = np.nanmin(all_y) - 2
            ymax = np.nanmax(all_y) + 1
            if flux:
                ax_abs.set_ylim(ymin, ymax)
            else:
                ax_abs.set_ylim(ymax, ymin)  # flip so bright = up

    if created_ax:
        ax_abs.set_title(f"ZTF Light Curve (intrinsic): {oid}", fontsize=16)
        ax_abs.set_xlabel("MJD", fontsize=14)
        if flux:
            ax_abs.set_ylabel("Flux (mJy)", fontsize=14)
        else:
            ax_abs.set_ylabel("Absolute magnitude", fontsize=14)
            if show_comparison and ax_app is not None:
                ax_app.set_ylabel("Apparent magnitude", fontsize=14)
                ax_app.invert_yaxis()
            ax_abs.invert_yaxis()
        if show_comparison and ax_app is not None:
            h1, l1 = ax_abs.get_legend_handles_labels()
            h2, l2 = ax_app.get_legend_handles_labels()
            ax_abs.legend(h1 + h2, l1 + l2, fontsize=9)
        else:
            ax_abs.legend(fontsize=9)
        ax_abs.grid(True, alpha=0.4)

        if show:
            plt.show()

    if not created_ax:
        return ax

def plot_forced_lc_abs_app(
    resdict, oid="ZTF source", ax=None, show=True, show_comparison=True,
    flux=False, xlim=(None, None), ylim=(None, None), SNU=5.0
    ):
    """
    Plot ZTF forced photometry intrinsic (absolute) light curve with apparent/absolute y-axes.

    Left y-axis: apparent magnitude
    Right y-axis: absolute magnitude (distance- and extinction-corrected)
    """


    if not any("mag_intrinsic" in d for d in resdict.values()):
        print("No corrected light curve data found. Run intrinsic_lc_corrections() first.")
        return
    
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        created_ax = True

    colors = {"ZTF_g": "green", "ZTF_r": "red", "ZTF_i": "orange"}
    markers = {"ZTF_g": "o", "ZTF_r": "X", "ZTF_i": "D"}

    all_mjd, all_y = [], []
    shown_limit_label = set()
    use_mJy = flux and any(("flux_mJy" in d) for d in resdict.values())

    # --- Plot data per filter ---
    for filt, data in resdict.items():
        color = colors.get(filt, "black")
        marker = markers.get(filt, "o")
        mjd = np.array(data["mjd"])

        # --- Detections ---
        mask_det = ~np.isnan(data["mag"])
        if np.any(mask_det):
            if flux:
                if "flux_mJy_intrinsic" in data:
                    yvals = data["flux_mJy_intrinsic"][mask_det]
                    yerrs = data["flux_err_mJy_intrinsic"][mask_det]
                else:
                    yvals = data["flux_intrinsic"][mask_det]
                    yerrs = data["flux_intrinsic_err"][mask_det]
            else:
                yvals = data["mag"][mask_det]  # apparent
                yerrs = data["mag_err"][mask_det]

            ax.errorbar(
                mjd[mask_det],
                yvals,
                yerr=yerrs,
                color=color,
                marker=marker,
                linestyle="none",
                label=filt,
            )

            all_mjd.extend(mjd[mask_det])
            all_y.extend(yvals)

        # --- Non-detections (limits) ---
        mask_nondet = np.isnan(data["mag_intrinsic"])
        if np.any(mask_nondet):
            if flux:
                if use_mJy and "lim_flux_intrinsic_mJy" in data:
                    yvals_lim = data["lim_flux_intrinsic_mJy"][mask_nondet]
                else:
                    yvals_lim = SNU * data["flux_intrinsic_err"][mask_nondet]
            else:
                if "mag_ul" in data:
                    yvals_lim = data["mag_ul"][mask_nondet]
                else:
                    yvals_lim = data.get(
                        "limiting_mag", np.full_like(mjd, np.nan)
                    )[mask_nondet]

            finite = np.isfinite(yvals_lim)
            ax.scatter(
                mjd[mask_nondet][finite],
                yvals_lim[finite],
                marker="v",
                alpha=0.5,
                color=color,
                s=40,
                label=f"lim. mag {filt}" if filt not in shown_limit_label else None,
            )
            shown_limit_label.add(filt)
            all_mjd.extend(mjd[mask_nondet][finite])
            all_y.extend(yvals_lim[finite])

    # --- Axis limits ---
    if all_mjd and all_y:
        if xlim[0] is not None and xlim[1] is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(np.nanmin(all_mjd) - 5, np.nanmax(all_mjd) + 5)

        if ylim[0] is not None and ylim[1] is not None:
            ax.set_ylim(ylim)
        else:
            ymin, ymax = np.nanmin(all_y) - 1, np.nanmax(all_y) + 1
            ax.set_ylim(ymax, ymin)  # invert for apparent mag (bright = up)

    # --- Labels and dual-axis setup ---
    if created_ax:
        ax.set_title(f"ZTF Light Curve (intrinsic): {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        if flux:
            ax.set_ylabel("Flux (mJy)", fontsize=14)
        else:
            ax.set_ylabel("Apparent magnitude", fontsize=14)
            ax.invert_yaxis()

            if show_comparison:
                ax_abs = ax.twinx()

                # --- Get distance modulus + extinction ---
                meta_path = Path(config.paths.zenodo_meta)
                meta = pd.read_csv(meta_path)
                meta_row = meta[meta["name"] == oid]
                if not meta_row.empty:
                    z = meta_row["redshift"].values[0]
                    dist_mod = calculate_distance_modulus(z)
                    Av_host = meta_row["Avhost"].values[0] if "Avhost" in meta_row.columns else 0.0
                    Av_MW = meta_row["A_V_MW"].values[0] if "A_V_MW" in meta_row.columns else 0.0
                    A_total = Av_host + Av_MW
                else:
                    dist_mod = 0
                    A_total = 0

                def app_to_abs(m): return m - dist_mod - A_total
                def abs_to_app(M): return M + dist_mod + A_total

                # Match apparent axis visually, just offset numerically
                app_ylim = ax.get_ylim()
                ax_abs.set_ylim(app_to_abs(app_ylim[0]), app_to_abs(app_ylim[1]))
                ax_abs.set_ylabel("Absolute magnitude", fontsize=14)
                ax.set_ylim(ymax, ymin) # same visual direction as left
                ax_abs.invert_yaxis()

        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()

        if show:
            plt.show()

    if not created_ax:
        return ax

##########################################################################
## -------------- WISE PHOTOMETRY PLOTTING FUNCTIONS ------------------ ##
##########################################################################

def plot_wise_lc(resdict, oid="ZTF source", xlim=(None, None), ax=None, show=True, subtract_parity_baseline=False, 
                 baseline_dt=200, show_baselines=False, clip_negatives=False):

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9,5))
        created_ax = True

    colors = {1:"navy",2:"dodgerblue"}
    markers = {1:"s",2:"s"}

    if subtract_parity_baseline:
        resdict = subtract_wise_parity_baseline(resdict, dt=baseline_dt, clip_negatives=clip_negatives)
        for fid, data in zip([1,2], [("b1_times","b1_fluxes","b1_fluxerrs"), ("b2_times","b2_fluxes","b2_fluxerrs")]):
            color = colors.get(fid,"black")

            # --- Detections ---
            mask_det = ~np.isnan(resdict[data[1]])
            if np.any(mask_det):
                ax.errorbar(
                    resdict[data[0]][mask_det],
                    resdict[data[1]][mask_det],
                    yerr=resdict[data[2]][mask_det],
                    fmt="o", color=color, label=f"W{fid}",
                    marker = markers.get(fid, "o"),
                )
                even_base = resdict[f"{data[0][:2]}_even_baseline"]
                odd_base = resdict[f"{data[0][:2]}_odd_baseline"]

                # optional show baselines on plot
                if show_baselines:
                    ax.axhline(even_base, color=color, linestyle='--', alpha=0.5, label=f"{fid} even baseline")
                    ax.axhline(odd_base, color=color, linestyle=':', alpha=0.5, label=f"{fid} odd baseline")

    else:
        for fid, data in zip([1,2], [("b1_times","b1_fluxes","b1_fluxerrs"), ("b2_times","b2_fluxes","b2_fluxerrs")]):
            color = colors.get(fid,"black")

            # --- Detections ---
            mask_det = ~np.isnan(resdict[data[1]])
            if np.any(mask_det):
                ax.errorbar(
                    resdict[data[0]][mask_det],
                    resdict[data[1]][mask_det],
                    yerr=resdict[data[2]][mask_det],
                    fmt="o", color=color, label=f"W{fid}",
                    marker = markers.get(fid, "o"),
                )

    if created_ax:
        ax.set_title(f"WISE Light Curve: {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        ax.set_ylabel("WISE flux (mJy)", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.4)

        if xlim[0] is not None and xlim[1] is not None:
            ax.set_xlim(xlim[0], xlim[1])
        
        if show:
            plt.show()

    if not created_ax:
        return ax

###########################################################################
######## ---------- COMBINED PLOTTING FUNCTIONS --------------- ###########
###########################################################################

def plot_combined_lc(ztf_resdict, wise_resdict, oid="ZTF+WISE source",
            xlim=(None, None), ztf_flux=False, mode="stacked", scale_wise=True,
            baseline_ref="ztf", baseline_dt=100, ref_band="r", logy=False,
            savepath=None, ax=None, labels=True, mark_tail_start=False,
            mark_plateau_end=False,  mark_custom_mjd=None
        ):   
    """
    Plot ZTF + WISE light curves.
    
    Parameters
    ----------
    ztf_resdict : dict
        ZTF results dictionary from get_ztf_lc_data().
    wise_resdict : dict
        WISE results dictionary from get_wise_lc_data().
    oid : str
        Object identifier for plot title.
    xlim : tuple
        (min_mjd, max_mjd) to set x-axis limits.
    ztf_flux : bool
        If True, plot ZTF in flux (mJy). If False, plot in magnitude.
    mode : str
        "stacked" → two panels (default).
        "overlay" → single panel with both ZTF and WISE.
    scale_wise : bool
        If True, scale WISE fluxes to match ZTF peak in reference band.
    ref_band : str
        Reference band for scaling WISE fluxes ("g", "r", or "i").
    """
    
    # Helper: log-safe clipping
    def log_safe(y, yerr=None):
        y = np.asarray(y, dtype=float)
        if np.all(~np.isfinite(y)) or np.all(y <= 0):
            return y, yerr
        positive = y[y > 0]
        floor = np.nanmin(positive) * 0.1
        y_clipped = np.where(y > 0, y, floor)
        if yerr is not None:
            # prevent errorbars from crossing below floor
            yerr = np.minimum(yerr, y_clipped * 0.95)
            return y_clipped, yerr
        return y_clipped, None
    
    # helper for v lines
    def vline_with_label(ax, x, label, *, y=0.02, dx_pts=-12,
                     line_kw=None, text_kw=None):
        line_kw = {} if line_kw is None else dict(line_kw)
        text_kw = {} if text_kw is None else dict(text_kw)

        ax.axvline(x, **line_kw)

        ax.annotate(
            label,
            xy=(x, y), xycoords=ax.get_xaxis_transform(),   # x=data, y=axes fraction
            xytext=(dx_pts, 0), textcoords="offset points", # fixed visual offset
            rotation=90,
            ha="center", va=("bottom" if y < 0.5 else "top"),
            clip_on=False,
            **text_kw
        )


    # x-axis range
    xlim=(ztf_resdict["lc_det"].mjd.min(), ztf_resdict["lc_det"].mjd.max())
    min_mjd = xlim[0] - 30 if xlim[0] is not None else None
    max_mjd = xlim[1] + 100 if xlim[1] is not None else None

    w = wise_resdict.copy()

    if mode == "stacked":
        # ---------------------------
        # Two-panel figure
        # ---------------------------
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10,8), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        # -----------------
        # Top panel: ZTF LC
        # -----------------
        if ztf_flux:
            ztf_resdict = convert_ZTF_mag_mJy(ztf_resdict, forced=True)

            # clip fluxes for log scale
            if logy:
                for band, data in ztf_resdict["forced"].items():
                    data["flux_mJy"], data["flux_err_mJy"] = log_safe(
                        data["flux_mJy"], data["flux_err_mJy"]
                    )

            plot_forced_lc(ztf_resdict['forced'], oid=ztf_resdict['oid'], xlim=(min_mjd, max_mjd), ax=ax1, show=False, flux=True)
            ax1.set_ylabel("ZTF flux (mJy)", fontsize=14)
            ax1.grid(True, alpha=0.4)
        else:
            plot_forced_lc(ztf_resdict['forced'], oid=ztf_resdict['oid'], xlim=(min_mjd, max_mjd), ax=ax1, show=False)
            ax1.set_ylabel("Apparent magnitude", fontsize=14)

        if logy and ztf_flux:
            ax1.set_yscale("log")
        
        ax1.legend()
        ax1.grid(True, alpha=0.4)

        # -----------------
        # Bottom panel: WISE LC
        # -----------------
        if wise_resdict == {}:
            pass
        else:

            if logy:
                w["b1_fluxes"], w["b1_fluxerrs"] = log_safe(w["b1_fluxes"], w["b1_fluxerrs"])
                w["b2_fluxes"], w["b2_fluxerrs"] = log_safe(w["b2_fluxes"], w["b2_fluxerrs"])

            plot_wise_lc(wise_resdict, oid=ztf_resdict['oid'], subtract_parity_baseline=scale_wise, xlim=(min_mjd, max_mjd), ax=ax2, show=False)
            ax2.set_ylabel("WISE flux scaled (mJy)" if scale_wise else "WISE flux (mJy)", fontsize=14)
            ax2.grid(True, alpha=0.4)
            ax2.set_xlabel("MJD", fontsize=14)

            if logy:
                ax2.set_yscale("log")

        # change style for stacked
        # 1. Fix scatter points (PathCollection)
        for ax in [ax1, ax2]:
            for col in ax.collections:
                if isinstance(col, mcoll.PathCollection):
                    fcs = col.get_facecolors()
                    if fcs is None or len(fcs) == 0:
                        continue
                    r, g, b, _ = fcs[0]
                    col.set_facecolor((r, g, b, 0.3))
                    col.set_edgecolor((r, g, b, 0.3))
                    col.set_linewidth(1.2)
                    col.set_sizes([20])

            # 2. Fix errorbar markers (Line2D)
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

        ax1.legend()
        ax2.legend()
        fig.subplots_adjust(hspace=0.0)
        fig.suptitle(f"ZTF + WISE Light Curve: {oid}", fontsize=16, y=0.93)

        if mark_tail_start and oid in m_dict:
            tail_start = m_dict[oid]
            for ax in (ax1, ax2):
                vline_with_label(
                    ax, tail_start, "Tail Start",
                    y=0.95, dx_pts=-10,
                    line_kw=dict(color="black", linestyle="--", alpha=0.7),
                    text_kw=dict(fontsize=9, color="black"),
                )

        if mark_plateau_end and oid in m_dict:
            plateau_end = m_dict[oid]
            for ax in (ax1, ax2):
                vline_with_label(
                    ax, plateau_end, f"Plateau End ({plateau_end:.1f})",
                    y=0.05, dx_pts=-10,
                    line_kw=dict(color="black", linestyle="-.", alpha=0.7),
                    text_kw=dict(fontsize=9, color="black"),
                )


        if mark_custom_mjd is not None:
            print(f"Marking custom MJD: {mark_custom_mjd}")
            for ax in (ax1, ax2):
                vline_with_label(
                    ax, mark_custom_mjd, f"SED ({mark_custom_mjd:.1f})",
                    y=0.2, dx_pts=-12,
                    line_kw=dict(color="red", alpha=0.7),
                    text_kw=dict(fontsize=9, color="black"),
                )



        if savepath:
            plt.savefig(savepath, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {savepath}")
        else:
            plt.show()

    elif mode == "overlay":
        # ---------------------------
        # Single-panel figure
        # ---------------------------
        created_ax = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
            created_ax = True
        else:
            fig = ax.figure

        # --- ZTF LC ---
        if ztf_flux:
            # print("Converting ZTF mag to mJy...")
            ztf_resdict = convert_ZTF_mag_mJy(ztf_resdict, forced=True)

            if logy:
                for band, data in ztf_resdict["forced"].items():
                    data["flux_mJy"], data["flux_err_mJy"] = log_safe(
                        data["flux_mJy"], data["flux_err_mJy"]
                    )

            plot_forced_lc(ztf_resdict['forced'], oid=ztf_resdict['oid'], xlim=(min_mjd, max_mjd), ax=ax, show=False, flux=True)
        else:
            plot_forced_lc(ztf_resdict['forced'], oid=ztf_resdict['oid'], xlim=(min_mjd, max_mjd), ax=ax, show=False)

        # --- WISE LC ---
        if wise_resdict == {}:
            pass
        else:

            if logy:
                w["b1_fluxes"], w["b1_fluxerrs"] = log_safe(w["b1_fluxes"], w["b1_fluxerrs"])
                w["b2_fluxes"], w["b2_fluxerrs"] = log_safe(w["b2_fluxes"], w["b2_fluxerrs"])

            plot_wise_lc(wise_resdict, oid=ztf_resdict['oid'], subtract_parity_baseline=scale_wise, xlim=(min_mjd, max_mjd), ax=ax, clip_negatives=True, show=False)


        # change style for overlay
        # 1. Fix scatter points (PathCollection)
        for col in ax.collections:
            if isinstance(col, mcoll.PathCollection):
                fcs = col.get_facecolors()
                if fcs is None or len(fcs) == 0:
                    continue
                r, g, b, _ = fcs[0]
                col.set_facecolor((r, g, b, 0.3))
                col.set_edgecolor((r, g, b, 0.3))
                col.set_linewidth(1.2)
                col.set_sizes([20])

        # 2. Fix errorbar markers (Line2D)
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

        if labels:
            ax.set_title(f"ZTF + WISE Light Curve: {oid}", fontsize=16)
            ax.set_xlabel("MJD", fontsize=14)
            ax.set_ylabel("Flux (mJy)", fontsize=14)
        
            handles, labels = ax.get_legend_handles_labels()
            # keep only ZTF detections and WISE bands
            keep = ["ZTF_g", "ZTF_r", "ZTF_i", "W1", "W2"]
            filtered = [(h, l) for h, l in zip(handles, labels) if l in keep]

            if filtered:
                handles, labels = zip(*filtered)
                ax.legend(handles, labels, loc="upper right")
        if not labels:
            ax.tick_params(labelbottom=True, labelleft=False)
            ax.set_xlabel("MJD", fontsize=10)

            # change marker size for inset
            for col in ax.collections:
                if isinstance(col, mcoll.PathCollection):
                    col.set_sizes([7])  

            # ---- resize errorbar markers ----
            for line in ax.lines:
                if line.get_marker() not in (None, "", "None"):
                    line.set_markersize(2) 


        ax.grid(True, alpha=0.4)

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0, max(ax.get_ylim()))

        if mark_tail_start:
            params_path = Path(config.paths.params)
            params = pd.read_csv(params_path)
            m = params[['name', 'plateauend', 'tailstart']].dropna()
            m_dict = dict(zip(m['name'].astype(str), m['tailstart'].astype(float)))
            if oid in m_dict:
                tail_start = m_dict[oid]
                if not labels:
                    ax.axvline(tail_start, color='black', linestyle='--', alpha=0.7, linewidth=1.3)
                else:
                    ax.axvline(tail_start, color='black', linestyle='--', alpha=0.7)
                if labels:
                    plateau_end = m_dict[oid]
                    vline_with_label(
                        ax, tail_start, "Tail Start",
                        y=0.02, dx_pts=-5,
                        line_kw=dict(color="black", linestyle="--", alpha=0.7),
                        text_kw=dict(fontsize=9, color="black"),
                    )
        
        if mark_plateau_end:
            params_path = Path(config.paths.params)
            params = pd.read_csv(params_path)
            m = params[['name', 'plateauend', 'tailstart']].dropna()
            m_dict = dict(zip(m['name'].astype(str), m['plateauend'].astype(float)))
            if oid in m_dict:
                plateau_end = m_dict[oid]
                if not labels:
                    ax.axvline(plateau_end, color='black', linestyle='-.', alpha=0.7, linewidth=1.3)
                else:
                    ax.axvline(plateau_end, color='black', linestyle='-.', alpha=0.7)
                if labels:
                    vline_with_label(
                        ax, plateau_end, "Plateau End",
                        y=0.02, dx_pts=-5,
                        line_kw=dict(color="black", linestyle="-.", alpha=0.7),
                        text_kw=dict(fontsize=9, color="black"),
                    )

        if mark_custom_mjd is not None:
            if labels:
                vline_with_label(
                            ax, mark_custom_mjd, f"SED MJD ({mark_custom_mjd:.0f})",
                            y=0.2, dx_pts=-10,
                            line_kw=dict(color="red", alpha=0.7, linewidth=3.5),
                            text_kw=dict(fontsize=10, color="black", fontweight='bold'),
                        )
            else:
                ax.axvline(mark_custom_mjd, color='red', alpha=0.7, linewidth=2.5, label=f"SED")
            

        if savepath:
            fig.savefig(savepath, bbox_inches="tight")
            print(f"Saved plot to {savepath}")
        elif created_ax:
            plt.show()

        return ax

    else:
        raise ValueError("Invalid mode. Choose 'stacked' or 'overlay'.")