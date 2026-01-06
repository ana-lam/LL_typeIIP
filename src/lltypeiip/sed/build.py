import numpy as np
import pandas as pd
import astropy.constants as const

from ..config import config, SNR_MIN, SNR_MIN_WISE, LAM_EFF
from ..photometry.wise import subtract_wise_parity_baseline


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
    Keep only SEDs that have at least two *detection* (not UL) in any ZTF band
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
                                     snr_min_wise=SNR_MIN_WISE, csv_path=config.paths.params,
                                     tail_offset_days=0.0, merge_dt=4.0, require_wise_detection=False,
                                     min_detected_bands=2, include_plateau_epoch=True):
    """
    Build SEDs for any epochs **after plateau end** that have >= `min_detected_bands`
    detections (regardless of whether they are ZTF or WISE).

    Parameters
    ----------
    ztf_resdict : dict
        ZTF results dictionary from get_ztf_lc_data().
    wise_resdict : dict
        WISE results dictionary from get_wise_lc_data().
    max_dt_ztf : float, optional
        Maximum time separation (days) for ZTF measurements. Default 4.0.
    max_dt_wise : float, optional
        Maximum time separation (days) for WISE measurements. Default 1.0.
    include_limits : bool, optional
        Whether to include upper limits in SEDs. Default True.
    snr_min : float, optional
        Minimum SNR for ZTF detections. If None, uses config.snr.min.
    snr_min_wise : float, optional
        Minimum SNR for WISE detections. If None, uses config.snr.min_wise.
    csv_path : str or Path, optional
        Path to parameters CSV file. If None, uses config.paths.params.
    tail_offset_days : float, optional
        Offset in days to apply to tail start time. Default 0.0.
    merge_dt : float, optional
        Merge epochs within this separation (days). Default 4.0.
    require_wise_detection : bool, optional
        If True, only build SEDs anchored on WISE detections. Default False.
    min_detected_bands : int, optional
        Minimum number of bands with detections. Default 2.
    include_plateau_epoch : bool, optional
        If True, include an SED at plateau end time. Default True.

    Returns
    -------
    list of dict
        List of SED dictionaries (same schema as build_sed).
    """

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
    
    Parameters
    ----------
    ztf_resdict : dict
        ZTF results dictionary from get_ztf_lc_data().
    wise_resdict : dict
        WISE results dictionary from get_wise_lc_data().
    max_dt_ztf : float, optional
        Maximum time separation (days) for ZTF measurements. Default 5.0.
    max_dt_wise : float, optional
        Maximum time separation (days) for WISE measurements. Default 5.0.
    include_limits : bool, optional
        Whether to include upper limits in SEDs. Default True.
    snr_min : float, optional
        Minimum SNR for ZTF detections. If None, uses config.snr.min.
    snr_min_wise : float, optional
        Minimum SNR for WISE detections. If None, uses config.snr.min_wise.
    
    Returns
    -------
    list of dict
        List of SED dictionaries.
    """

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
    
    w1_t = _det_times(w.get("b1_times", []), w.get("b1_fluxes", []), w.get("b1_fluxerrs", []), snr_min_wise)
    w2_t = _det_times(w.get("b2_times", []), w.get("b2_fluxes", []), w.get("b2_fluxerrs", []), snr_min_wise)

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
    Build a single-epoch SED from ZTF and WISE photometry.
    
    Parameters
    ----------
    mjd0 : float
        Reference epoch (MJD) for the SED.
    ztf_resdict : dict
        ZTF results dictionary with 'forced' key containing photometry.
    wise_resdict : dict
        WISE results dictionary from get_wise_lc_data().
    max_dt_ztf : float, optional
        Maximum time separation (days) for ZTF measurements. Default 1.0.
    max_dt_wise : float, optional
        Maximum time separation (days) for WISE measurements. Default 1.0.
    include_limits : bool, optional
        Whether to include upper limits. Default True.
    snr_min : float, optional
        Minimum SNR for ZTF detections. If None, uses config.snr.min.
    snr_min_wise : float, optional
        Minimum SNR for WISE detections. If None, uses config.snr.min_wise.
    
    Returns
    -------
    dict
        SED dictionary with keys: mjd, oid, bands, nu, lam, Fnu, eFnu, is_ul, dt_labels.
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
        lam = LAM_EFF[band]
        nu = (const.c.value / (lam * 1e-10))  # Hz
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
        lam = LAM_EFF[b]
        nu = (const.c.value / (lam * 1e-10))  # Hz
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