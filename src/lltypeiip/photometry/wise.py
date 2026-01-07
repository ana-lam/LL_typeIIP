import numpy as np
import glob
import json
from pathlib import Path
from ..config import config

def subtract_wise_parity_baseline(wise_resdict, clip_negatives=False, dt=200, 
                                  rescale_uncertainties=True, NEOWISE_t0=56650.0,
                                  sigma_clip=3.0, verbose=False, phase_aware=True):
    """
    Subtract separate baselines for even/odd WISE epochs to account
    for scan-orientation systematics. Also clips negative/zero fluxes
    for safe log-scale plotting. And rescales uncertainties by sqrt(<reduced_chisq>).

    Parameters
    ----------
    wise_resdict : dict
        Dictionary with keys like 'b1_times', 'b1_fluxes', etc.
    dt : float
        Days before the peak to compute baseline.

    Returns
    -------
    w : dict
        Copy of input dictionary with corrected fluxes and stored baselines.
    """

    # --- RMS for S/N ---
    def _robust_rms_p84_p16(x):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        p16, p84 = np.percentile(x, [16, 84])
        return 0.5*(p84 - p16)

    if wise_resdict == {}:
        return {}
    w = wise_resdict.copy()

    for band in ["b1", "b2"]:
        times = np.array(w[f"{band}_times"])
        fluxes = np.array(w[f"{band}_fluxes"])
        flux_errs = np.array(w[f"{band}_fluxerrs"])

        if len(fluxes) == 0:
            continue

        # --- find burst (peak flux) ---
        peak_idx = np.nanargmax(fluxes)
        t_peak = times[peak_idx]

        # --- even vs odd indices ---
        even_idx = np.arange(len(fluxes)) % 2 == 0
        odd_idx  = ~even_idx

        # --- phase masks ------
        m_p1 = (times < NEOWISE_t0)
        m_p2 = (times >= NEOWISE_t0) & (times < (t_peak - dt))
        m_post = (times >= (t_peak - dt))

        # --- compute baselines pre-peak-dt ---
        if phase_aware:
            # phase 1
            e_base_p1 = np.nanmedian(fluxes[m_p1 & even_idx]) if np.any(m_p1 & even_idx) else 0.0
            o_base_p1 = np.nanmedian(fluxes[m_p1 & odd_idx]) if np.any(m_p1 & odd_idx) else 0.0
            # phase 2
            e_base_p2 = np.nanmedian(fluxes[m_p2 & even_idx]) if np.any(m_p2 & even_idx) else 0.0
            o_base_p2 = np.nanmedian(fluxes[m_p2 & odd_idx]) if np.any(m_p2 & odd_idx) else 0.0
            # create per-epoch baseline to subtract
            per_epoch_baseline = np.zeros_like(fluxes, dtype=float)
            
            per_epoch_baseline[m_p1 & even_idx] = e_base_p1
            per_epoch_baseline[m_p1 & odd_idx] = o_base_p1

            per_epoch_baseline[m_p2 & even_idx] = e_base_p2
            per_epoch_baseline[m_p2 & odd_idx] = o_base_p2

            per_epoch_baseline[m_post & even_idx] = e_base_p2
            per_epoch_baseline[m_post & odd_idx] = o_base_p2

            even_base, odd_base = e_base_p2, o_base_p2

        else:
            pre = (times >= NEOWISE_t0) & (times < (t_peak - dt))
            even_base = np.nanmedian(fluxes[pre & even_idx]) if np.any(pre & even_idx) else 0.0
            odd_base  = np.nanmedian(fluxes[pre & odd_idx]) if np.any(pre & odd_idx) else 0.0
            per_epoch_baseline = np.zeros_like(fluxes, dtype=float)
            per_epoch_baseline[pre & even_idx] = even_base
            per_epoch_baseline[pre & odd_idx] = odd_base

        # --- subtract parity-specific baselines ---
        corrected_fluxes = fluxes.copy()
        corrected_fluxes -= per_epoch_baseline

        # ---- rescale uncertainties by chi_sq->1 ----
        scale_bg = 1.0
        chi2_red = np.nan
        scale_rms = 1.0
        snr_rms_rob = np.nan

        if rescale_uncertainties:
            # ----- Step 1: compute reduced chi^2 of background points and rescale -----
            # background window: [background_start, t_sn_start)
            bg_mask_time = m_p2

            if phase_aware:
                res = np.empty_like(fluxes, dtype=float)
                res[:] = np.nan
                res[bg_mask_time & even_idx] = fluxes[bg_mask_time & even_idx]-e_base_p2
                res[bg_mask_time & odd_idx] = fluxes[bg_mask_time & odd_idx]-o_base_p2
            else:
                res = corrected_fluxes.copy()

            use = bg_mask_time & (flux_errs > 0) & ~np.isnan(res)
            if np.any(use) and np.isfinite(sigma_clip) and sigma_clip > 0:
                z = np.zeros_like(res, dtype=float)
                z[:] = np.nan
                z[use] = res[use] / flux_errs[use]
                use = use & (np.abs(z) <= sigma_clip)

            n_bg = int(sum(use))

            k = (1 if np.any(bg_mask_time & even_idx) else 0) + (1 if np.any(bg_mask_time & odd_idx) else 0) # number of fitted parameters 
            dof = max(n_bg - k, 1) 

            if n_bg >= (k+1): # need at least k+1 points to estimate variance
                chi2 = np.nansum((res[use]/flux_errs[use])**2)
                chi2_red = chi2 / dof
                
                if verbose:
                    print(f"{band.upper()} background points: {n_bg}, dof={dof}, chi2_red={chi2_red:.2f}")
                
                if np.isfinite(chi2_red) and chi2_red > 0:
                    scale_bg = np.sqrt(chi2_red)
                    flux_errs *= scale_bg
                    if verbose:
                        print(f"Rescaling {band.upper()} flux uncertainties by {scale_bg:.2f}")

            if np.isfinite(chi2_red) and chi2_red > 0 and np.any(use):
                chi2_red = np.nansum((res[use]/flux_errs[use])**2) / dof
                if verbose:
                    print(f"Post-rescaling {band.upper()} chi2_red={chi2_red:.2f}")

            # ----- Step 2: compute RMS of background points and rescale -----
            # robust RMS estimator (p84-p16)/2
            snr = corrected_fluxes[bg_mask_time] / flux_errs[bg_mask_time]
            valid = np.isfinite(snr)

            if np.sum(valid) >= 3: # need at least 3 points to estimate RMS
                snr_rms_rob = _robust_rms_p84_p16(snr[valid]) # 0.5*(P84 - P16)
                if np.isfinite(snr_rms_rob) and snr_rms_rob > 0:
                    scale_rms = snr_rms_rob
                    flux_errs *= scale_rms
                    if verbose:
                        print(f"Rescaling {band.upper()} flux uncertainties S/N RMS = {snr_rms_rob:.3f} by {scale_rms:.2f}")
            
            if np.sum(valid) >= 3 and np.isfinite(snr_rms_rob):
                snr2 = corrected_fluxes[bg_mask_time] / flux_errs[bg_mask_time]
                snr2_rms_rob = _robust_rms_p84_p16(snr2[valid])
                if verbose:
                    print(f"Post-rescaling {band.upper()} S/N RMS = {snr2_rms_rob:.3f}")

        # --- clip negatives/zeros for log plotting ---
        if clip_negatives:
            positive_mask = corrected_fluxes > 0
            if np.any(positive_mask):
                flux_floor = np.nanmin(corrected_fluxes[positive_mask]) * 0.1
                corrected_fluxes[~positive_mask] = flux_floor
            else:
                # edge case: all values ≤ 0
                corrected_fluxes[:] = 1e-4

        # --- store results ---
        w[f"{band}_fluxes"] = corrected_fluxes
        w[f"{band}_fluxerrs"] = flux_errs
        w[f"{band}_even_baseline"] = even_base
        w[f"{band}_odd_baseline"] = odd_base
        w[f"{band}_chisq_red"] = chi2_red if np.isfinite(chi2_red) else np.nan
        w[f"{band}_uncertainty_scale"] = scale_bg
        w[f"{band}_snr_rms_rob"] = float(snr_rms_rob) if np.isfinite(snr_rms_rob) else np.nan
        w[f"{band}_uncertainty_scale_rms"] = float(scale_rms)
        if verbose:
            print(f"{band.upper()} baselines: even={even_base:.4f}, odd={odd_base:.4f}")

    return w

def get_wise_lc_data(oid):

    try:
        filename = glob.glob(f"{config.paths.wise_dir}/lightcurve_{oid}_*.json")[0]
    except IndexError:
        print(f"No WISE light curve file found for {oid}")
        return {}

    f = open(filename, 'r')
    jfile = json.load(f)
    outmags = [jfile[j] for j in jfile.keys()]
    times = np.array([o['mjd'] for o in outmags])
    fluxes = np.array([o['psfflux'] for o in outmags])
    fluxerrs = np.array([o['psfflux_unc'] for o in outmags])
    bands = np.array([o['bandid'] for o in outmags])
    zps = np.array([o['zp'] for o in outmags])
    zpflux = np.zeros(len(bands))
    #create zero point fluxes in Jy as provided in the WISE official release supplements
    for i in range(len(bands)):
        if bands[i] == 1:
            zpflux[i] = 309
        else:
            zpflux[i] = 172
    zps = np.array([o['zp'] for o in outmags])
    mjy_fluxes = zpflux * 10**(-zps/2.5) * fluxes * 1e3
    mjy_fluxerrs = zpflux * 10**(-zps/2.5) * fluxerrs * 1e3
    snrs = fluxes/fluxerrs
    
    b1filt = (bands == 1)
    b2filt = (bands == 2)
    
    b1_times = times[b1filt]
    b1_fluxes = mjy_fluxes[b1filt]
    b1_fluxerrs = mjy_fluxerrs[b1filt]
    
    b2_times = times[b2filt]
    b2_fluxes = mjy_fluxes[b2filt]
    b2_fluxerrs = mjy_fluxerrs[b2filt]    
    
    resdict = {'b1_times': b1_times, 'b1_fluxes': b1_fluxes, 'b1_fluxerrs': b1_fluxerrs,
               'b2_times': b2_times, 'b2_fluxes': b2_fluxes, 'b2_fluxerrs': b2_fluxerrs}
    return resdict