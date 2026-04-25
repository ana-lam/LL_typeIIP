import numpy as np
import pandas as pd
import astropy.constants as const

from ..config import config, SNR_MIN, SNR_MIN_WISE, LAM_EFF
from ..photometry.wise import subtract_wise_parity_baseline
from ..photometry.ztf import estimate_texp_mjd_from_forced

def _unwrap_sed(obj):
    """Unwrap SED dict"""
    if isinstance(obj, dict) and "sed" in obj and isinstance(obj["sed"], dict):
        sed = obj["sed"]
        if "phase_days" not in sed and "phase" in obj:
            sed["phase_days"] = obj["phase"]
        return sed
    return obj

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

def _wise_anchored_epochs(wise_resdict, t_tail, snr_min_wise=SNR_MIN_WISE, merge_dt=1.0, first_wise_only=False):
    """
    Build candidate SED spochs anchored on WISE detection times.

    For each WISE epoch, mjd0 is the median of W1/W@ detection times that fall
    within merge_dt of each other. If first_wise_only is True, only use the earliest 
    WISE detection time as the candidate epoch. Only consider WISE detections that occur
    after t_tail.

    """

    w1_t = _det_times(wise_resdict.get("b1_times", []), wise_resdict.get("b1_fluxes", []),
                      wise_resdict.get("b1_fluxerrs", []), snr_min_wise)
    w2_t = _det_times(wise_resdict.get("b2_times", []), wise_resdict.get("b2_fluxes", []),
                      wise_resdict.get("b2_fluxerrs", []), snr_min_wise)
    
    w1_t = w1_t[w1_t > t_tail]
    w2_t = w2_t[w2_t > t_tail]

    all_wise = np.unique(np.concatenate([w1_t, w2_t])) if (w1_t.size or w2_t.size) else np.array([])

    if all_wise.size == 0:
        return np.array([])

    epochs = _merge_epochs(all_wise, merge_dt=merge_dt)

    if first_wise_only and len(epochs) > 0:
        return np.array([epochs[0]])
    
    return epochs

def _trim_ztf_detections(sed, max_n_det=4, trim_min_dt=3.0):
    """
    If SED has more than max_n_det *detections*, drop ZTF detections one at a time in
    descending order of |Δt|, if |Δt| > trim_min_dt. 
    """
    bands = np.array(sed["bands"])
    is_ul = np.array(sed["is_ul"])
    dt_labels = np.array(sed["dt_labels"])

    # count current detections
    n_det = sum(1 for ul in is_ul if not ul)
    if n_det <= max_n_det:
        return sed  # no trimming can be performed

    ztf_det_indices = []
    for i, (band, ul, dt_label) in enumerate(zip(bands, is_ul, dt_labels)):
        if band.startswith("ZTF") and not ul:
            try:
                dt_val = abs(float(dt_label.split("=")[1].split(" ")[0]))
            except (IndexError, ValueError):
                dt_val = 0.0
            ztf_det_indices.append((i, dt_val))
    
    # only candidates |Δt| > trim_min_dt can be trimmed
    can_trim = [(i, dt) for i, dt in ztf_det_indices if dt > trim_min_dt]

    if not can_trim:
        return sed  # no candidates for trimming
    
    can_trim.sort(key=lambda x: x[1], reverse=True)  # sort by descending |Δt|

    n_to_drop = n_det - max_n_det
    n_ztf_det = len(ztf_det_indices)
    n_to_drop = min(n_to_drop, len(can_trim), n_ztf_det - 1) 

    drop_indices = set(idx for idx, _ in can_trim[:n_to_drop])
    if not drop_indices:
        return sed  # no indices to drop

    print(f"  Trimming {len(drop_indices)} ZTF detection(s) with |Δt| > {trim_min_dt} d: "
          f"{[dt_labels[i] for i in sorted(drop_indices)]}")
    
    keep = [i for i in range(len(bands)) if i not in drop_indices]
    
    for key in ("bands", "nu", "lam", "Fnu", "eFnu", "is_ul", "dt_labels"):
        sed[key] = [sed[key][i] for i in keep]

    return sed


# --- Build SEDs after tail-onset using tail start from CSV -----
def build_multi_epoch_seds_from_tail(ztf_resdict, wise_resdict, max_dt_ztf=4.0, 
                                     max_dt_wise=1.0, include_limits=True, snr_min=SNR_MIN,
                                     snr_min_wise=SNR_MIN_WISE, csv_path=config.paths.params,
                                     tail_offset_days=0.0, merge_dt=1.0, require_wise_detection=False,
                                     min_detected_bands=2, include_plateau_epoch=True,
                                     first_wise_only=False, trim_to_n_det=None, trim_min_dt=3.0):
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
    first_wise_only : bool, optional
        If True, only build SEDs anchored on the first WISE detection time. Default False.
    trim_to_n_det : int, optional
        If set, trim SEDs to have at most this number of ZTF detections.
    trim_min_dt : float, optional
        Minimum time separation (days) for trimming ZTF detections. Default 3.0.

    Returns
    -------
    list of dict
        List of SED dictionaries (same schema as build_sed).
    """

    oid = ztf_resdict.get("oid")
    ztf_forced = ztf_resdict['forced']

    params_df = pd.read_csv(csv_path)
    m = params_df[['name', 'plateauend', 'tailstart']].dropna()
    if include_plateau_epoch:
        m_dict_plateauend = dict(zip(m['name'].astype(str), m['plateauend'].astype(float))) 
        m_dict_tailstart = dict(zip(m['name'].astype(str), m['tailstart'].astype(float))) 
        delta_days = m_dict_tailstart[oid] - m_dict_plateauend[oid] if oid in m_dict_plateauend and oid in m_dict_tailstart else None
        if delta_days > 0:
            halfway = delta_days / 2
            t_tail = m_dict_plateauend[oid] + halfway
        else:
            t_tail = m_dict_tailstart[oid] if oid in m_dict_tailstart else None
        print("T_TAIL ACTUAL OFFSET FROM TAIL START:", m_dict_tailstart[oid]-t_tail, f"(tailstart-t_tail, tailstart={m_dict_tailstart[oid]}, t_tail={t_tail})")
    else:
        m_dict = dict(zip(m['name'].astype(str), m['tailstart'].astype(float))) # use tail start directly
        t_tail = m_dict[oid] if oid in m_dict else None

    if t_tail is None:
        return []


    t_tail = t_tail + float(tail_offset_days) # shift tail start time/plateau end time

    # ---- candidate epochs from WISE ----
    
    w = subtract_wise_parity_baseline(
        wise_resdict, clip_negatives=False, dt=200.0,
        rescale_uncertainties=True, sigma_clip=3.0
    )

    all_epochs = _wise_anchored_epochs(wise_resdict, t_tail, snr_min_wise=snr_min_wise,
                                            merge_dt=merge_dt, first_wise_only=first_wise_only)
    
    if all_epochs.size == 0:
        print(f". No WISE detections found after tail start (MJD={t_tail:.1f}) for {oid}")
        return []
    
    print(f"  WISE-anchored candidate epochs for {oid}: {all_epochs}")

    # --- explosion time estimate ---
    t_exp_info = estimate_texp_mjd_from_forced(ztf_forced)
    t_exp_mjd = t_exp_info['t_exp_mjd'] if t_exp_info else None
    t_exp_sig = t_exp_info['sigma_mjd'] if t_exp_info else None
    t_exp_band = t_exp_info['band'] if t_exp_info else None

    # --- build SEDs ---
    seds = []
    for mjd0 in all_epochs:
        sed = build_sed(mjd0, ztf_resdict, wise_resdict,
                        max_dt_ztf=max_dt_ztf, max_dt_wise=max_dt_wise,
                        include_limits=include_limits, snr_min=snr_min,
                        snr_min_wise=snr_min_wise)
        sed["wise_anchor_mjd"] = float(mjd0)
        sed["t_exp_mjd"] = t_exp_mjd
        sed["t_exp_sig"] = t_exp_sig
        sed["t_exp_band"] = t_exp_band
        sed["phase_days"] = (mjd0 - t_exp_mjd) if t_exp_mjd is not None else np.nan

        # --- optional trimming of ZTF detections ---
        if trim_to_n_det is not None:
            sed = _trim_ztf_detections(sed, max_n_det=trim_to_n_det, trim_min_dt=trim_min_dt)

        # --- quality flags ---
        ztf_dts = []
        for band, ul, label in zip(sed["bands"], sed["is_ul"], sed["dt_labels"]):
            if band.startswith("ZTF") and not ul:
                try:
                    dt_val = float(label.split("=")[1].split(" ")[0])
                    ztf_dts.append(dt_val)
                except (IndexError, ValueError):
                    pass
        sed["max_dt_ztf_actual"] = float(np.max(np.abs(ztf_dts))) if ztf_dts else np.nan

        if sed["bands"] and _sed_has_required_detections(sed,
                                require_wise_detetection=require_wise_detection,
                                min_detected_bands=min_detected_bands):
            seds.append(sed)
            print(f"  Accepted SED at MJD={mjd0:.2f} | bands={sed['bands']} "
                  f"| max_dt_ztf_actual={sed['max_dt_ztf_actual']:.2f} d")
            print("SED MJD ACTUAL OFFSET FROM TAIL START:", m_dict_tailstart[oid] - mjd0, f"(tailstart-mjd0, tailstart={m_dict_tailstart[oid]}, mjd0={mjd0})")

    return seds

    # # ---- candidate epochs from ZTF -----
    # if include_plateau_epoch:
    #     # include plateau end epoch as well
    #     all_epochs = np.array([t_tail])
    
    # ztf_det_times = []

    # for band in ["ZTF_g","ZTF_r","ZTF_i"]:
    #     if band in ztf_forced:
    #         d = ztf_forced[band]
    #         t_band = _det_times(d["mjd"], d["flux_mJy"], d["flux_err_mJy"], snr_min)
    #         ztf_det_times.append(t_band[t_band > t_tail])

    # ztf_det_times = [np.asarray(a, float) for a in ztf_det_times if a is not None and len(a) > 0]
    # ztf_det_times = np.unique(np.concatenate(ztf_det_times)) if ztf_det_times else np.array([])

    # all_epochs = _merge_epochs(ztf_det_times, merge_dt=merge_dt)

    # # ---- candidate epochs from WISE ----

    # ######## USE WISE DET TO ANCHOR MJD0 SELECTION ########

    # wise_det_times = []

    # w = subtract_wise_parity_baseline(
    #     wise_resdict, clip_negatives=False, dt=200.0,
    #     rescale_uncertainties=True, sigma_clip=3.0
    # )


    # w1_t = _det_times(w.get("b1_times", []), w.get("b1_fluxes", []), w.get("b1_fluxerrs", []), snr_min_wise)
    # w2_t = _det_times(w.get("b2_times", []), w.get("b2_fluxes", []), w.get("b2_fluxerrs", []), snr_min_wise)

    # if require_wise_detection:
    #     all_epochs = np.array([])  # reset to only WISE times

    # if w1_t.size:
    #     wise_det_times.append(w1_t[w1_t > t_tail])
    #     print("WISE W1 det times:", w1_t[w1_t > t_tail])
    # if w2_t.size:
    #     wise_det_times.append(w2_t[w2_t > t_tail])
    #     print("WISE W2 det times:", w2_t[w2_t > t_tail])


    # wise_det_times = np.unique(np.concatenate(wise_det_times) if wise_det_times else np.array([]))
    # print("wise det times:", wise_det_times)
    # combined_det_times = np.unique(np.concatenate([all_epochs, wise_det_times])) if all_epochs.size and wise_det_times.size else all_epochs if all_epochs.size else wise_det_times
    # print(combined_det_times)
    # all_epochs = _merge_epochs(combined_det_times, merge_dt=merge_dt)

    # # estimate explosion time from forced photometry
    # t_exp_info = estimate_texp_mjd_from_forced(ztf_forced)
    # t_exp_mjd = t_exp_info['t_exp_mjd'] if t_exp_info else None
    # t_exp_sig = t_exp_info['sigma_mjd'] if t_exp_info else None
    # t_exp_band = t_exp_info['band'] if t_exp_info else None

    # # ---- build SEDs -----
    # seds = []
    # print("all epochs:", all_epochs)
    # for mjd0 in all_epochs:
    #     sed = build_sed(mjd0, ztf_resdict, wise_resdict,
    #                     max_dt_ztf=max_dt_ztf, max_dt_wise=max_dt_wise,
    #                     include_limits=include_limits, snr_min=snr_min,
    #                     snr_min_wise=snr_min_wise)
        
    #     sed["t_exp_mjd"] = t_exp_mjd
    #     sed["t_exp_sig"] = t_exp_sig
    #     sed["t_exp_band"] = t_exp_band
    #     sed["phase_days"] = (mjd0 - t_exp_mjd) if t_exp_mjd is not None else np.nan

    #     if sed["bands"] and _sed_has_required_detections(sed, 
    #                                                      require_wise_detetection=require_wise_detection, 
    #                                                      min_detected_bands=min_detected_bands):
    #         seds.append(sed)
            

    # return seds

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
        elif include_limits:
            ul = _nearest_ul(times, errs, mjd0, max_dt_wise, n_sigma=3)
            if ul:
                t_ul, f_ul = ul
                sed["bands"].append(b)
                sed["nu"].append(nu)
                sed["lam"].append(lam)
                sed["Fnu"].append(f_ul)
                sed["eFnu"].append(np.nan)
                sed["is_ul"].append(True)
                sed["dt_labels"].append(f"Δt={t_ul-mjd0:+.2f} d")

    return sed

# --- Prepare x, y for plotting with units ----
def _prepare_sed_xy(sed, y_mode="Fnu"):
    """
    y_mode:
      'Fnu'  -> x = nu [Hz],    y = Fnu [mJy]
      'Flam' -> x = lam [micron],      y = lambda*Flam [erg s^-1 cm^-2]
               (i.e., lambdaF_lambda with lambda on the x-axis in micrometers)
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
        # Compute lambdaF_lambda directly from F_nu using: lambdaF_lambda = (c / lambda) * F_nu
        # Units:
        #   F_nu (mJy) -> cgs: 1 mJy = 1e-26 erg s^-1 cm^-2 Hz^-1
        #   c in cgs: 2.99792458e10 cm/s
        #   lambda in cm: 1 Angstroms = 1e-8 cm
        Fnu_cgs   = Fnu  * 1e-26  # erg s^-1 cm^-2 Hz^-1
        eFnu_cgs  = eFnu * 1e-26
        lam_cm    = lam * 1e-8 # cm
        lamF      = (const.c.to('cm/s').value / lam_cm) * Fnu_cgs     # erg s^-1 cm^-2
        e_lamF    = (const.c.to('cm/s').value / lam_cm) * eFnu_cgs
        
        # x-axis in micrometers (microns): 1 microns = 10,000 Angstroms
        x  = lam * 1e-4 # microns
        y  = lamF
        ey = e_lamF

        x_label = r"$\lambda\ \mathrm{(\mu m)}$"
        y_label = r"$\lambda F_\lambda\ \mathrm{(erg\ cm^{-2}\ s^{-1})}$"
    
    else:
        raise ValueError("y_mode must be 'Fnu' or 'Flam'.")

    return x, y, ey, x_label, y_label