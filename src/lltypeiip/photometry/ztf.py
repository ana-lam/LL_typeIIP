import numpy as np
import pandas as pd
import pickle
import time
from io import StringIO
from pathlib import Path
from astroquery.irsa import Irsa
from requests.auth import HTTPBasicAuth
import os
from dotenv import load_dotenv

from ..config import config, SNR_MIN

# IRSA FP
IRSA_FP_URL = "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"
IRSA_FP_GET = "https://ztfweb.ipac.caltech.edu/cgi-bin/getForcedPhotometry.cgi"

load_dotenv()
username = os.getenv("IRSA_USERNAME")
password = os.getenv("IRSA_PASSWORD")

# create session with IRSA credentials
if username and password:
    Irsa._session = HTTPBasicAuth(username, password)

# Mag constants
AB_MAGNITUDE_ZEROPOINT = 3631  # Jy (for AB magnitude system)
MAG_TO_FLUX_FACTOR = 2.5 / np.log(10)  # 1.0857

## Using a lot of Viraj's code from LSST CCA Summer School

def query_all_detections(oid, client, page_size=10000):
    """Fetch all detection rows for a given oid from ALeRCE."""
    all_dets = []
    page = 1
    while True:
        dets = client.query_detections(
            oid, format='pandas', page=page, page_size=page_size
        )
        if dets.empty:
            break
        all_dets.append(dets)
        page += 1
    return pd.concat(all_dets, ignore_index=True) if all_dets else pd.DataFrame()


def query_all_nondetections(oid, client, page_size=10000):
    """Fetch all non-detection rows for a given oid from ALeRCE."""
    all_nd = []
    page = 1
    while True:
        nd = client.query_non_detections(
            oid, format='pandas', page=page, page_size=page_size
        )
        if nd.empty:
            break
        all_nd.append(nd)
        page += 1
    return pd.concat(all_nd, ignore_index=True) if all_nd else pd.DataFrame()

def get_ztf_forcedphot(filename, SNT=SNR_MIN, SNU=5.0): # SNT & SNU values from https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_forced_photometry.pdf
    """
    Parse a ZTF forced photometry .dat file into times, fluxes, errors, and filters.
    Returns a dictionary grouped by filter.
    """
    # Load into pandas, skipping header lines beginning with '#'
    df = pd.read_csv(
        filename,
        comment='#',
        sep=r"\s+",
        header=0,
        skiprows=1
    )
    df.columns = df.columns.str.replace(",", "").str.strip()

    # Convert JD to MJD
    df["mjd"] = df["jd"] - 2400000.5

    # Compute SNR
    flux = df['forcediffimflux'].astype(float)
    flux_err = df['forcediffimfluxunc'].astype(float)
    snr = flux / flux_err

    # Detections
    det = (snr > SNT) & (flux > 0)

    # Compute magnitudes and errors for detections only
    # mag = zpdiff - 2.5*log10(flux), valid only if flux > 0
    df["mag"] = np.nan
    df["mag_err"] = np.nan
    df.loc[det, "mag"] = df.loc[det, "zpdiff"] - 2.5 * np.log10(df.loc[det, "forcediffimflux"])
    df.loc[det, "mag_err"] = MAG_TO_FLUX_FACTOR * df.loc[det, "forcediffimfluxunc"] / df.loc[det, "forcediffimflux"]

    # Non-detections (upper limits)
    # mag_UL = zpdiff - 2.5*log10(SNU * sigma_flux)
    non_det = ~det
    df['mag_ul'] = np.nan
    df.loc[non_det, "mag_ul"] = df.loc[non_det, "zpdiff"] - 2.5 * np.log10(SNU * df.loc[non_det, "forcediffimfluxunc"])

    # Organize output
    resdict = {}
    for filt in df["filter"].unique():
        mask = df["filter"] == filt
        resdict[filt] = {
            "mjd": df.loc[mask, "mjd"].values,
            "flux": df.loc[mask, "forcediffimflux"].values, # DN units
            "flux_err": df.loc[mask, "forcediffimfluxunc"].values, # DN units
            "snr": snr.loc[mask].values, 
            "mag": df.loc[mask, "mag"].values,
            "mag_err": df.loc[mask, "mag_err"].values,
            "mag_ul": df.loc[mask, "mag_ul"].values,
            "limiting_mag": df.loc[mask, "diffmaglim"].values,
        }
    return resdict

def get_ztf_lc_data(oid, client, ra=None, dec=None,
                    doLC=False, doStamps=False, add_forced=False,
                    pad_before=100, pad_after=600):
    """
    Fetch detections, non-detections, and optionally IRSA forced photometry for a ZTF object.

    Parameters
    ----------
    oid : str
        Object ID.
    client : ALeRCE client
        ALeRCE API client instance.
    ra, dec : float, optional
        Sky coordinates (deg, ICRS). Required if add_forced=True.
    doLC : bool, default False
        Whether to plot ALeRCE light curve.
    doStamps : bool, default False
        Whether to fetch and plot image stamps.
    add_forced : bool, default False
        Whether to query IRSA forced photometry service.
    pad_before, pad_after : int
        Days before and after the first detection to include in forced photometry request.
    """

    results = {"oid": oid}

    # --- Fetch ALeRCE detections
    try:
        lc_det = client.query_detections(oid, format='pandas').sort_values("mjd")
        results["lc_det"] = lc_det
    except Exception as e:
        print(f"Could not fetch detections for {oid}: {e}")
        lc_det = pd.DataFrame()

    # --- Fetch ALeRCE non-detections ---
    try:
        lc_nondet = client.query_non_detections(oid, format='pandas').sort_values("mjd")
        results["lc_nondet"] = lc_nondet
    except Exception as e:
        print(f"Could not fetch non-detections for {oid}: {e}")
        lc_nondet = pd.DataFrame()

    # --- Plot ALeRCE light curve ---
    if doLC and not lc_det.empty and not lc_nondet.empty:
        from .plotting import plot_lc
        plot_lc(oid, lc_det, lc_nondet)

    # --- Plot image stamps ---
    if doStamps and not lc_det.empty:
        from .plotting import plot_stamps
        plot_stamps(oid, lc_det, client)

    if add_forced:
        # Use Kaustav's published forced photometry data
        forced_phot_dir = Path(config.paths.forced_photometry_dir)
        forced_file = forced_phot_dir / f"{oid}_fps.dat"
        
        if forced_file.exists():
            res_forced = get_ztf_forcedphot(str(forced_file))
            results['forced'] = res_forced
        else:
            print(f"Warning: Forced photometry file not found: {forced_file}")

    # Save results
    output_dir = Path(config.paths.data_dir) / "ztf_alerce"
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_filename = output_dir / f"{oid}.pkl"
    
    with open(pkl_filename, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved ztf_resdict for {oid} to {output_dir}")
    return results

def convert_ZTF_mag_mJy(res, forced=False):
    """
    Convert ZTF magnitudes to mJy flux units.
    
    Parameters
    ----------
    res : dict
        ZTF results dictionary with lc_det, lc_nondet, and optionally forced keys.
    forced : bool, optional
        If True, also convert forced photometry data. Default False.
    
    Returns
    -------
    dict
        Modified results dictionary with flux_mJy and flux_err_mJy fields added.
    
    Notes
    -----
    Uses AB magnitude system: F_nu[mJy] = 3631 * 10^(-mag/2.5) * 1000
    """
    
    # --- Detections ---
    if "lc_det" in res and not res["lc_det"].empty:
        mask_det = res["lc_det"].magpsf.notna()
        mag = res["lc_det"].loc[mask_det, "magpsf"].values
        mag_err = res["lc_det"].loc[mask_det, "sigmapsf"].values

        flux_mJy = AB_MAGNITUDE_ZEROPOINT * 10**(-mag/2.5) * 1e3
        flux_err_mJy = flux_mJy * MAG_TO_FLUX_FACTOR * mag_err

        res["lc_det"].loc[mask_det, "flux_mJy"] = flux_mJy
        res["lc_det"].loc[mask_det, "flux_err_mJy"] = flux_err_mJy 
        
    # --- Non-detections ---
    if "lc_nondet" in res and not res["lc_nondet"].empty:
        mask_nondet = res["lc_nondet"].diffmaglim > 0
        lim_mag = res["lc_nondet"].loc[mask_nondet, "diffmaglim"].values
        lim_flux_mJy = AB_MAGNITUDE_ZEROPOINT * 10**(-lim_mag/2.5) * 1e3
        res["lc_nondet"].loc[mask_nondet, "lim_flux_mJy"] = lim_flux_mJy

    # --- Forced photometry ---
    if forced and "forced" in res:
        for filt, data in res["forced"].items():
            # initialize if missing
            if "flux_mJy" not in data:
                data["flux_mJy"] = np.full_like(data["mag"], np.nan, dtype=float)
                data["flux_err_mJy"] = np.full_like(data["mag"], np.nan, dtype=float)
                data["lim_flux_mJy"] = np.full_like(data["limiting_mag"], np.nan, dtype=float)

            # detections
            mask_det = ~np.isnan(data["mag"])
            mag = data["mag"][mask_det]
            mag_err = data["mag_err"][mask_det]
            flux_mJy = AB_MAGNITUDE_ZEROPOINT * 10**(-mag/2.5) * 1e3
            flux_err_mJy = flux_mJy * MAG_TO_FLUX_FACTOR * mag_err
            data["flux_mJy"][mask_det] = flux_mJy
            data["flux_err_mJy"][mask_det] = flux_err_mJy

            # non-detections: prefer mag_ul, fallback to limiting_mag
            has_ul = "mag_ul" in data
            mask_nondet = np.isnan(data["mag"]) & (
                np.isfinite(data["mag_ul"]) if has_ul else (data["limiting_mag"] > 0)
            )
            if np.any(mask_nondet):
                lim_mag = (data["mag_ul"][mask_nondet] if has_ul
                           else data["limiting_mag"][mask_nondet])
                lim_flux_mJy = AB_MAGNITUDE_ZEROPOINT * 10**(-lim_mag/2.5) * 1e3
                data["lim_flux_mJy"][mask_nondet] = lim_flux_mJy
    
    return res 

### ---- QUERY IRSA FOR FORCED PHOTOMETRY DATA -----

def submit_forced_phot_irsa(ra, dec, jdstart, jdend, email=None):
    """Submit a forced photometry job to IRSA using astroquery's authenticated session."""
    payload = {
        "ra": ra,
        "dec": dec,
        "jdstart": jdstart,
        "jdend": jdend,
    }
    if email:
        payload["email"] = email
    r = Irsa._session.post(IRSA_FP_URL, data=payload)
    print(r.status_code)
    r.raise_for_status()

    for line in r.text.splitlines():
        if "request id" in line.lower():
            return line.split()[-1]
    raise RuntimeError("Could not parse job ID from response:\n" + r.text)

def fetch_forced_phot_irsa(jobid, wait=30, maxtries=20):
    """Poll IRSA until results are ready, using astroquery session."""
    for _ in range(maxtries):
        r = Irsa._session.get(f"{IRSA_FP_GET}?pid={jobid}")
        if "is still being processed" in r.text:
            print("Still processing... waiting", wait, "s")
            time.sleep(wait)
            continue

        # If job done, IRSA returns a .dat file
        df = pd.read_csv(StringIO(r.text), comment="#", delim_whitespace=True)
        return df

    raise TimeoutError("Forced photometry request did not finish in time.")

def parse_forced_df(df):
    """Convert IRSA forced photometry DataFrame into your resdict structure."""
    df["mjd"] = df["jd"] - 2400000.5
    df["mag"] = np.nan
    df["mag_err"] = np.nan
    good = df["forcediffimflux"] > 0
    df.loc[good, "mag"] = df.loc[good, "zpdiff"] - 2.5*np.log10(df.loc[good, "forcediffimflux"])
    df.loc[good, "mag_err"] = 1.0857 * df.loc[good, "forcediffimfluxunc"] / df.loc[good, "forcediffimflux"]

    resdict = {}
    for filt in df["filter"].unique():
        mask = df["filter"] == filt
        resdict[filt] = {
            "mjd": df.loc[mask, "mjd"].values,
            "flux": df.loc[mask, "forcediffimflux"].values,
            "flux_err": df.loc[mask, "forcediffimfluxunc"].values,
            "mag": df.loc[mask, "mag"].values,
            "mag_err": df.loc[mask, "mag_err"].values,
            "limiting_mag": df.loc[mask, "diffmaglim"].values,
        }
    return resdict

def estimate_texp_mjd_from_forced(res_forced, snr_det=5.0, snr_nd=2.0,
                                  follwup_window=5.0, min_followup_dets=2,
                                  prefer_filters=['ZTF_r', 'ZTF_g', 'ZTF_i']):
    """
    Estimate explosion time from forced photometry.
    - Find first detection above snr_det and flux > 0 and require
        it is part of a sustained rise (min_followup_dets within follwup_window days in the same band).
    - Find last non-detection before that time with snr < snr_nd in the same band.
    - Estimate t_exp as midpoint between last non-detection and first detection
        t_exp = 0.5 * (t_nd + t_det)
    """

    best = None
    
    for band in prefer_filters:
        if band not in res_forced:
            continue
        data = res_forced[band]
        mjd = data['mjd']
        flux = data['flux']
        flux_err = data['flux_err']
        
        mask = np.isfinite(mjd) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)

        if not np.any(mask):
            continue

        mjd, flux, flux_err = mjd[mask], flux[mask], flux_err[mask]

        snr = flux / flux_err

        order = np.argsort(mjd)
        mjd, flux, flux_err, snr = mjd[order], flux[order], flux_err[order], snr[order]

        det_mask = (snr >= snr_det) & (flux > 0)
        if not np.any(det_mask):
            continue

        # find first detection with sustained rise
        det_idx = np.where(det_mask)[0]
        t_det = None
        for idx in det_idx:
            t0 = mjd[idx]
            window_mask = (mjd >= t0) & (mjd <= t0 + follwup_window)
            n_followup_dets = np.sum(det_mask & window_mask)
            if n_followup_dets >= min_followup_dets:
                t_det = t0
                break

        if t_det is None:
            continue

        pre_mask = mjd < t_det
        if not np.any(pre_mask):
            continue

        nd_mask = (snr <= snr_nd) & pre_mask
        if np.any(nd_mask):
            t_nd = mjd[np.where(nd_mask)[0][-1]]
        else:
            t_nd = mjd[np.where(pre_mask)[0][-1]]

        dt = t_det - t_nd
        if dt <= 0:
            continue

        cand = dict(band=band, t_nd_mjd=float(t_nd), t_det_mjd=float(t_det),
                    t_exp_mjd=float(0.5 * (t_nd + t_det)), sigma_mjd=float(0.5*dt),
                    dt=float(dt))
    
        if best is None or cand['dt'] < best['dt']:
            best = cand

    return best