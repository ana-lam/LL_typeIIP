import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
import astropy.constants as const
from lc_plotting_functions import subtract_wise_parity_baseline
import pandas as pd

# effective wavelengths [Angstrom]
lam_eff = {
    "ZTF_g": 4770.0,
    "ZTF_r": 6231.0,
    "ZTF_i": 7625.0,
    "W1": 34000.0,
    "W2": 46000.0,
}

SNR_MIN = 3.0 # minimum SNR for a detection
SNR_MIN_WISE = 2.0 # minimum SNR for a WISE detection (slightly more relaxed)

SED_COLORS  = {"ZTF_g":"green", "ZTF_r":"red", "ZTF_i":"orange", "W1":"navy", "W2":"dodgerblue"}
SED_MARKERS = {"ZTF_g":"o", "ZTF_r":"X", "ZTF_i":"D", "W1":"s", "W2":"s"}
SED_SIZE = 20  # dot area


# --- Helper functions for selecting nearest points and building SEDs ---
def _pick_nearest(time_mjd, val_mJy, err_mJy, mjd0, max_dt, snr_min=SNR_MIN,
                  band=None, snr_min_wise=None, require_positive_flux=True):
    """
    Pick the nearest-in-time detection with S/N >= snr_min and positive flux.
    Parameters
    ----------
    time_mjd : array(float)
    val_mJy, err_mJy : Quantity arrays with unit mJy
    Returns (t, f, e) with f,e as Quantities or None.
    """
    if len(time_mjd) == 0:
        return None
    dt = np.abs(time_mjd - mjd0)
    order = np.argsort(dt)

    thresh = snr_min
    if band in ["W1","W2"] and snr_min_wise is not None:
        thresh = snr_min_wise

    for k in order:
        if dt[k] > max_dt:
            break
        f = (val_mJy[k])
        e = (err_mJy[k])
        if not (np.isfinite(f) and np.isfinite(e) and e>0):
            continue
        if require_positive_flux and f <= 0:
            continue
        if (f/e) >= thresh:
            return (time_mjd[k], f, e)

    return None

def _nearest_ul(time_mjd, err_mJy, mjd0, max_dt, n_sigma=3):
    """
    If no detection, provide an n_sigma upper limit at the nearest time (within window).
    Returns (t_ul, F_ul) with F_ul as Quantity or None.
    """
    if len(time_mjd) == 0:
        return None
    k = np.argmin(np.abs(time_mjd - mjd0))
    if np.abs(time_mjd[k]-mjd0) <= max_dt and np.isfinite(err_mJy[k]) and err_mJy[k]>0:
        return (time_mjd[k], n_sigma*err_mJy[k])
    return None

def _sed_has_required_detections(sed, require_wise_detetection=True,
                                 min_detected_bands=2):
    """
    Keep only SEDs that have at least one *detection* (not UL) in any ZTF band
    AND at least one *detection* in any WISE band.
    """

    bands = np.array(sed["bands"])
    is_ul = np.array(sed["is_ul"])

    # detections mask
    det = ~is_ul
    det_bands = bands[det]

    # any ZTF detection?
    any_ztf_det = np.any(det & np.isin(bands, ["ZTF_g", "ZTF_r", "ZTF_i"]))
    # any WISE detection?
    any_wise_det = np.any(det & np.isin(bands, ["W1", "W2"]))

    # drop SEDs with no detections at all (i.e., only ULs)
    any_detection = np.any(det)

    n_det_bands = np.unique(det_bands).size

    if require_wise_detetection:
        return any_detection and any_ztf_det and any_wise_det and (n_det_bands >= min_detected_bands)
    else:
        return any_detection and any_ztf_det and (n_det_bands >= min_detected_bands)

# --- Build SEDs after tail-onset using tail start from CSV -----
def build_multi_epoch_seds_from_tail(ztf_resdict, wise_resdict, max_dt_ztf=4.0, 
                                     max_dt_wise=1.0, include_limits=True, snr_min=SNR_MIN,
                                     snr_min_wise=SNR_MIN_WISE, csv_path="data/TableA_full_machine_readable_params.csv",
                                     tail_offset_days=0.0, merge_dt=4.0, require_wise_detection=False,
                                     min_detected_bands=2, include_plateau_epoch=True):
    """
    Build SEDs for any epochs **after plateau end** that have >= `min_detected_bands`
    detections (regardless of whether they are ZTF or WISE).

    Returns
    -------
    list of SED dicts (same schema as build_sed)
    """

    # Because we loop over WISE detections, we can yield the same epoch so let's dedup
    def _merge_epochs(times, merge_dt=1.0):
        """
        Merge epochs that are within merge_dt days of each other.
        """

        if len(times) == 0:
            return np.array([])
        times = np.sort(times)
        groups = [[times[0]]]
        for x in times[1:]:
            if x - groups[-1][-1] <= merge_dt:
                groups[-1].append(x)
            else:
                groups.append([x])

        reps = [np.median(g) for g in groups]

        return np.array(reps)
    
    def _det_times(times, fluxes, errs, snr_threshold):
        """
        Find detection times based on SNR threshold.
        """
        t = np.asarray(times, float)
        f = np.asarray(fluxes, float)
        e = np.asarray(errs, float)
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (f > 0) & ((f / e) >= snr_threshold)
        
        return t[ok]

    oid = ztf_resdict.get("oid")
    ztf_forced = ztf_resdict['forced']

    params_df = pd.read_csv(csv_path)
    m = params_df[['name', 'plateauend', 'tailstart']].dropna()
    m_dict = dict(zip(m['name'].astype(str), m['plateauend'].astype(float))) # use plateau end for tail start

    if oid not in m_dict or not np.isfinite(m_dict[oid]):
        return []
    t_tail = float(m_dict[oid]) + float(tail_offset_days) # shift tail start time/plateau end time

    # ---- candidate epochs from ZTF -----
    if include_plateau_epoch:
        # include plateau end epoch as well
        all_epochs = np.array([t_tail])
    
    ztf_det_times = []

    for band in ["ZTF_g","ZTF_r","ZTF_i"]:
        if band in ztf_forced:
            d = ztf_forced[band]
            t_band = _det_times(d["mjd"], d["flux_mJy"], d["flux_err_mJy"], snr_min)
            ztf_det_times.append(t_band[t_band > t_tail])

    ztf_det_times = [np.asarray(a, float) for a in ztf_det_times if a is not None and len(a) > 0]
    ztf_det_times = np.unique(np.concatenate(ztf_det_times)) if ztf_det_times else np.array([])

    all_epochs = _merge_epochs(ztf_det_times, merge_dt=merge_dt)

    # ---- candidate epochs from WISE ----

    ######## USE WISE DET TO ANCHOR MJD0 SELECTION ########

    wise_det_times = []

    w = subtract_wise_parity_baseline(
        wise_resdict, clip_negatives=False, dt=200.0,
        rescale_uncertainties=True, sigma_clip=3.0
    )


    w1_t = _det_times(w.get("b1_times", []), w.get("b1_fluxes", []), w.get("b1_fluxerrs", []), snr_min_wise)
    w2_t = _det_times(w.get("b2_times", []), w.get("b2_fluxes", []), w.get("b2_fluxerrs", []), snr_min_wise)

    if require_wise_detection:
        all_epochs = np.array([])  # reset to only WISE times

    if w1_t.size:
        wise_det_times.append(w1_t[w1_t > t_tail])
    if w2_t.size:
        wise_det_times.append(w2_t[w2_t > t_tail])


    wise_det_times = np.unique(np.concatenate(wise_det_times) if wise_det_times else np.array([]))
    combined_det_times = np.unique(np.concatenate([all_epochs, wise_det_times])) if all_epochs.size and wise_det_times.size else all_epochs if all_epochs.size else wise_det_times
    all_epochs = _merge_epochs(combined_det_times, merge_dt=merge_dt)

    # ---- build SEDs -----
    seds = []
    for mjd0 in all_epochs:
        sed = build_sed(mjd0, ztf_resdict, wise_resdict,
                        max_dt_ztf=max_dt_ztf, max_dt_wise=max_dt_wise,
                        include_limits=include_limits, snr_min=snr_min,
                        snr_min_wise=snr_min_wise)
        if sed["bands"] and _sed_has_required_detections(sed, 
                                                         require_wise_detetection=require_wise_detection, 
                                                         min_detected_bands=min_detected_bands):
            seds.append(sed)
            

    return seds

# --- Build SEDs after ZTF peak -----
def build_multi_epoch_seds(ztf_resdict, wise_resdict, max_dt_ztf=5.0, max_dt_wise=5.0,
                           include_limits=True, snr_min=SNR_MIN, snr_min_wise=SNR_MIN_WISE):
    """
    Build SEDs for all WISE epochs after the ZTF SN peak,
    requiring ZTF+WISE coverage within a 5-day window.
    """

    # Because we loop over WISE detections, we can yield the same epoch so let's dedup
    def _merge_epochs(times, merge_dt=1.0):
        """
        Merge epochs that are within merge_dt days of each other.
        """

        if len(times) == 0:
            return np.array([])
        times = np.sort(times)
        groups = [[times[0]]]
        for x in times[1:]:
            if x - groups[-1][-1] <= merge_dt:
                groups[-1].append(x)
            else:
                groups.append([x])

        reps = [np.median(g) for g in groups]

        return np.array(reps)


    ztf_forced = ztf_resdict['forced']

    # ---- find ZTF peak in r-band ----
    if "ZTF_r" in ztf_forced:
        peak_idx = np.nanargmax(ztf_forced["ZTF_r"]["flux_mJy"])
        t_peak = ztf_forced["ZTF_r"]["mjd"][peak_idx]
    else:
        raise ValueError("ZTF_r data required to determine peak time.")
    
    # ---- candidate epochs from WISE ----
    w = subtract_wise_parity_baseline(
        wise_resdict, clip_negatives=False, dt=200.0,
        rescale_uncertainties=True, sigma_clip=3.0
    )

    # helper: keep only real detections per (relaxed) WISE threshold
    def _good_times(times, fluxes, errs, thr):
        t = np.asarray(times, float); f = np.asarray(fluxes, float); e = np.asarray(errs, float)
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e) & (e > 0) & (f > 0) & (f/e >= thr)
        return t[ok]
    
    w1_t = _good_times(w.get("b1_times", []), w.get("b1_fluxes", []), w.get("b1_fluxerrs", []), snr_min_wise)
    w2_t = _good_times(w.get("b2_times", []), w.get("b2_fluxes", []), w.get("b2_fluxerrs", []), snr_min_wise)

    all_epochs = np.unique(np.concatenate([w1_t[w1_t > t_peak], w2_t[w2_t > t_peak]]))
    all_epochs = _merge_epochs(all_epochs)

    # ---- build SEDs ----
    seds = []
    for mjd0 in all_epochs:
        sed = build_sed(mjd0, ztf_resdict, wise_resdict,
                        max_dt_ztf=max_dt_ztf, max_dt_wise=max_dt_wise,
                        include_limits=include_limits, snr_min=snr_min,
                        snr_min_wise=snr_min_wise)
        if sed["bands"] and _sed_has_required_detections(sed):
            seds.append(sed)
    
    return seds


def build_sed(mjd0, ztf_resdict, wise_resdict, max_dt_ztf=1.0, max_dt_wise=1.0, 
              include_limits=True, snr_min=SNR_MIN, snr_min_wise=SNR_MIN_WISE):
    """
    ztf_forced: dict with keys "ZTF_g/r/i" each containing arrays:
        'mjd', 'flux_mJy', 'flux_err_mJy' (numbers)
    wise_resdict: output of your WISE parser (numbers in mJy)
    """

    sed = {
        "mjd": mjd0,
        "bands": [],
        "nu": [],    # Quantity Hz
        "lam": [],   # Quantity Angstrom
        "Fnu": [],   # Quantity mJy
        "eFnu": [],  # Quantity mJy (nan for ULs)
        "is_ul": [],
        "dt_labels": [],
    }

    sed["oid"] = ztf_resdict.get("oid")
    ztf_forced = ztf_resdict['forced']

    # --- WISE, baseline removed without clipping ---
    w = subtract_wise_parity_baseline(
            wise_resdict, clip_negatives=False, dt=200.0,
            rescale_uncertainties=True, sigma_clip=3.0
        )
    wise_map = {"W1": ("b1_times","b1_fluxes","b1_fluxerrs"),
                "W2": ("b2_times","b2_fluxes","b2_fluxerrs")}

    # ZTF (forced, difference flux; assumes your dict has flux_mJy & flux_err_mJy)
    
    for band in ["ZTF_g","ZTF_r","ZTF_i"]:
        if band not in ztf_forced: 
            continue
        d = ztf_forced[band]
        tsel = _pick_nearest(
            np.asarray(d["mjd"], float),
            np.asarray(d["flux_mJy"], float),
            np.asarray(d["flux_err_mJy"], float),
            mjd0, max_dt_ztf,
            snr_min=snr_min, band=band,
            snr_min_wise=snr_min_wise
        )
        lam = lam_eff[band]
        nu = (const.c.value / (lam * 1e-10)) # Hz
        if tsel:
            t, f, e = tsel
            sed["bands"].append(band)
            sed["nu"].append(nu)
            sed["lam"].append(lam)
            sed["Fnu"].append(f)
            sed["eFnu"].append(e)
            sed["is_ul"].append(False)
            sed["dt_labels"].append(f"Δt={t-mjd0:+.2f} d")
        elif include_limits:
            ul = _nearest_ul(d["mjd"], d["flux_err_mJy"], mjd0, max_dt_ztf, 3)
            if ul:
                t_ul, f_ul = ul
                sed["bands"].append(band)
                sed["nu"].append(nu)
                sed["lam"].append(lam)
                sed["Fnu"].append(f_ul)
                sed["eFnu"].append(np.nan)
                sed["is_ul"].append(True)
                sed["dt_labels"].append(f"Δt={t_ul-mjd0:+.2f} d (3σ UL)")

    # WISE
    for b in ["W1","W2"]:
        tkey, fkey, ekey = wise_map[b]
        times = np.asarray(w[tkey], dtype=float)
        fluxes = np.asarray(w[fkey], dtype=float)
        errs = np.asarray(w[ekey], dtype=float)
        tsel = _pick_nearest(times, fluxes, errs, mjd0, max_dt_wise,
                             snr_min=snr_min, band=b, snr_min_wise=snr_min_wise)
        lam = lam_eff[b]
        nu = (const.c.value / (lam * 1e-10)) # Hz
        if tsel:
            t, f, e = tsel
            sed["bands"].append(b)
            sed["nu"].append(nu)
            sed["lam"].append(lam)
            sed["Fnu"].append(f)
            sed["eFnu"].append(e)
            sed["is_ul"].append(False)
            sed["dt_labels"].append(f"Δt={t-mjd0:+.2f} d")

    return sed

# --- Prepare x, y for plotting with units ----
def _prepare_sed_xy(sed, y_mode="Fnu"):
    """
    y_mode:
      'Fnu'  -> x = nu [Hz],    y = Fnu [mJy]
      'Flam' -> x = lam [micron],      y = lambda*Flam [erg s^-1 cm^-2]
               (i.e., λF_λ with λ on the x-axis in micrometers)
    """

    nu   = np.asarray(sed["nu"],  float)
    lam  = np.asarray(sed["lam"], float)
    Fnu  = np.asarray(sed["Fnu"], float)
    eFnu = np.asarray(sed["eFnu"], float)

    if y_mode == "Fnu":
        x = nu
        y = Fnu
        ey = eFnu

        x_label = r"$\nu\ \mathrm{(Hz)}$"
        y_label = r"$F_\nu\ \mathrm{(mJy)}$"

    elif y_mode == "Flam":
        # Compute λF_λ directly from F_ν using: λF_λ = (c / λ) * F_ν
        # Units:
        #   F_ν (mJy) -> cgs: 1 mJy = 1e-26 erg s^-1 cm^-2 Hz^-1
        #   c in cgs: 2.99792458e10 cm/s
        #   λ in cm: 1 Å = 1e-8 cm
        Fnu_cgs   = Fnu  * 1e-26  # erg s^-1 cm^-2 Hz^-1
        eFnu_cgs  = eFnu * 1e-26
        lam_cm    = lam * 1e-8 # cm
        lamF      = (const.c.to('cm/s').value / lam_cm) * Fnu_cgs     # erg s^-1 cm^-2
        e_lamF    = (const.c.to('cm/s').value / lam_cm) * eFnu_cgs
        
        # x-axis in micrometers (μm): 1 μm = 10,000 Å
        x  = lam * 1e-4 # μm
        y  = lamF
        ey = e_lamF

        x_label = r"$\lambda\ \mathrm{(\mu m)}$"
        y_label = r"$\lambda F_\lambda\ \mathrm{(erg\ cm^{-2}\ s^{-1})}$"
    
    else:
        raise ValueError("y_mode must be 'Fnu' or 'Flam'.")

    return x, y, ey, x_label, y_label

# --------- plotter ----------
def plot_sed(sed, ax=None, y_mode ="Fnu", logy=False, logx=False, title_prefix="SED", 
             secax=False, savepath=None):
    """
    y_mode='Fnu'  -> Fnu vs nu (mJy, Hz)
    y_mode='Flam' -> Flam vs λ (cgs/Ang, Ang)
    """

    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))
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
    ax.set_title(f"{sed['oid']}: {title_prefix} near MJD {sed['mjd']:.2f}", fontsize=13)
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
        ax.legend(H, L, fontsize=9)
    
    
    plt.tight_layout()

    if savepath:
            plt.savefig(savepath, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {savepath}")
    
    if created_ax:
        plt.show()

    return ax