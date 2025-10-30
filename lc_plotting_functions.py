import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from alerce.core import Alerce
from tqdm import tqdm
import requests
import time
import json
import glob
import os
from astroquery.irsa import Irsa
# from astroquery.irsa_dust import IrsaDust
from dustmaps.sfd import SFDQuery
from dotenv import load_dotenv
from io import StringIO
from astropy.stats import sigma_clipped_stats
from astropy.cosmology import Planck18 as cosmo
from astropy import coordinates as coord
from astropy import units as u
from requests.auth import HTTPBasicAuth
import matplotlib.collections as mcoll
from dust_extinction.parameter_averages import G23

##########################################################################
## ---------- ZTF LIGHT CURVE DATA PARSING AND PLOTTING --------------- ##
##########################################################################

IRSA_FP_URL = "https://ztfweb.ipac.caltech.edu/cgi-bin/requestForcedPhotometry.cgi"
IRSA_FP_GET = "https://ztfweb.ipac.caltech.edu/cgi-bin/getForcedPhotometry.cgi"

load_dotenv()
username = os.getenv("IRSA_USERNAME")
password = os.getenv("IRSA_PASSWORD")

if username is None or password is None:
    raise RuntimeError("Missing IRSA credentials. Set IRSA_USERNAME and IRSA_PASSWORD in your .env file.")

# Create a session with your IRSA credentials
Irsa._session = requests.Session()
Irsa._session.auth = HTTPBasicAuth(username, password)

# metadata df
meta = pd.read_csv("data/zenodo_metasample.csv")

import dustmaps.config
dustmaps.config.config['data_dir'] = '/Users/ana/Documents/LL_typeIIP/dustmaps_data'
sfd = SFDQuery(map_dir='/Users/ana/Documents/LL_typeIIP/dustmaps_data/sfd')

## Using a lot of Viraj's code from LSST CCA Summer School

# Color config for filters
colors = {1: "green", 2: "red", 3: "orange"}
labels = {1: 'g', 2: 'r', 3: 'i'}
markers = {1: 'o', 2: 'X', 3: 'D'}
sizes = {1: 30, 2: 60, 3: 90}

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

def plot_stamps(oid, lc_det, client):
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
    fig, ax = plt.subplots(figsize=(9, 5))

    # Loop over whatever filters are actually present
    for fid in sorted(SN_det.fid.dropna().unique()):
        color = colors.get(fid, "black")
        label = labels.get(fid, f"fid={fid}")
        marker = markers.get(fid, "o")
        size = sizes.get(fid, 40)

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


def get_ztf_forcedphot(filename, SNT=3.0, SNU=5.0): # SNT & SNU values from https://irsa.ipac.caltech.edu/data/ZTF/docs/ztf_forced_photometry.pdf
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
    # 1.0857 = 2.5 / ln(10)
    df.loc[det, "mag_err"] = 1.0857 * df.loc[det, "forcediffimfluxunc"] / df.loc[det, "forcediffimflux"]

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
        plot_lc(oid, lc_det, lc_nondet)

    # --- Plot image stamps ---
    if doStamps and not lc_det.empty:
        plot_stamps(oid, lc_det, client)

    # --- Fetch forced photometry from IRSA ---
    if add_forced:

        # Use Kaustav's published forced photometry data
        forced_file = f"data/Das_forced_photometry_files/{oid}_fps.dat"
        res_forced = get_ztf_forcedphot(forced_file)
        results['forced'] = res_forced

    return results

def convert_ZTF_mag_mJy(res, forced=False):
    """
    Convert ZTF magnitudes to mJy.
    """
    
    # --- Detections ---
    if "lc_det" in res and not res["lc_det"].empty:
        mask_det = res["lc_det"].magpsf.notna()
        mag = res["lc_det"].loc[mask_det, "magpsf"].values
        mag_err = res["lc_det"].loc[mask_det, "sigmapsf"].values

        flux_mJy = 3631 * 10**(-mag/2.5) * 1e3
        flux_err_mJy = flux_mJy * (np.log(10)/2.5) * mag_err

        res["lc_det"].loc[mask_det, "flux_mJy"] = flux_mJy
        res["lc_det"].loc[mask_det, "flux_err_mJy"] = flux_err_mJy 
        
    # --- Non-detections ---
    if "lc_nondet" in res and not res["lc_nondet"].empty:
        mask_nondet = res["lc_nondet"].diffmaglim > 0
        lim_mag = res["lc_nondet"].loc[mask_nondet, "diffmaglim"].values
        lim_flux_mJy = 3631 * 10**(-lim_mag/2.5) * 1e3
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
            flux_mJy = 3631 * 10**(-mag/2.5) * 1e3
            flux_err_mJy = flux_mJy * (np.log(10)/2.5) * mag_err
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
                lim_flux_mJy = 3631 * 10**(-lim_mag/2.5) * 1e3
                data["lim_flux_mJy"][mask_nondet] = lim_flux_mJy
    
    return res 

##########################################################################
## ------ CORRECTIONS FOR INTRINSIC LIGHT CURVE FROM ZTF DATA --------- ##
##########################################################################

def _add_MW_extinction(meta_df, coords_df_path="data/ZTF_coords.csv"):
    """
    Add Milky Way extinction E(B-V) and A_V,MW to metadata DataFrame.
    Following Schlafly & Finkbeiner (2011) scaling (0.86 * SFD),
    Cardelli et al. (1989) law, and R_V = 3.1.
    """

    sfd = SFDQuery()
    R_V = 3.1
    SF11_SCALE = 0.86
    
    EBV_list = []
    A_V_MW_list = []
    coords_df = pd.read_csv(coords_df_path)

    for name in meta_df["name"]:
        row_coords = coords_df[coords_df['oid'] == name]
        if row_coords.empty:
            EBV = np.nan
            A_V_MW = np.nan
        else:
            ra = float(row_coords['meanra'].values[0])
            dec = float(row_coords['meandec'].values[0])
            c = coord.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            EBV = float(sfd(c)) * SF11_SCALE
            A_V_MW = R_V * EBV
        EBV_list.append(EBV)
        A_V_MW_list.append(A_V_MW)

    meta_df['EBV_MW'] = EBV_list
    meta_df['A_V_MW'] = A_V_MW_list
    meta_df.to_csv("data/zenodo_metasample_with_MW.csv", index=False)

def calculate_distance_modulus(z):
    """Calculate distance modulus from redshift using Planck18 cosmology."""
    dl_pc = cosmo.luminosity_distance(z).to('pc').value
    return 5 * np.log10(dl_pc / 10)

def correct_extinction(mag, filt, Av_MW, Av_host=0.0):
    """
    Correct magnitude for Milky Way and host galaxy extinction.

    Parameters
    ----------
    mag : float, array
        Observed magnitude.
    filt : str
        Filter name.
    Av_host : float, optional
        Host galaxy V-band extinction. Default is 0.0.
    Av_MW: float
        Milky Way V-band extinction.

    Returns
    -------
    mag_corr : float, array 
        Extinction-corrected magnitude.
    A_total : float
        Total extinction applied (MW + host).
    """

    # Band-dependent A_lambda / A_V ratios for ZTF (~PS1 system, RV=3.1)
    coeffs = {
        "g": 3.172 / 2.742,  # 1.157
        "r": 2.271 / 2.742,  # 0.828
        "i": 1.682 / 2.742,  # 0.613
    }
    f = filt.lower()[0]
    factor = coeffs.get(f, 1.0)  # fallback to 1.0 if unknown filter

    # total extinction in this band
    A_total = factor * (Av_MW + Av_host)

    mag_corr = mag - A_total
    return mag_corr, A_total

def intrinsic_lc_corrections(res, sigma_A=True, flux=True):
    """
    Apply distance and extinction corrections to ZTF light curve data."""

    meta_df = meta

    if "forced" not in res:
        print("No forced photometry data to correct.")
        return res
    
    oid = res["oid"]
    forced = res["forced"].copy()

    meta_row = meta_df[meta_df['name']==res['oid']]
    if meta_row.empty:
        print(f"No metadata found for {res['oid']}.")
        return res
    
    # --- Distance modulus ---
    z = meta_row['redshift'].values[0]
    dist_mod = calculate_distance_modulus(z)

    # -- Extinction parameters ---
    _add_MW_extinction(meta_df)
    Av_host = meta_row['Avhost'].values[0] if 'Avhost' in meta_row.columns else 0.0
    Av_MW = meta_row['A_V_MW'].values[0] if 'A_V_MW' in meta_row.columns else 0.0

    # --- Optional uncertainties ----
    if sigma_A:
        sigma_A = 0.05
    else:
        sigma_A = 0.0

    # --- Apply corrections per filter ---
    for filt, data in forced.items():
        if "mag" not in data or np.all(np.isnan(data["mag"])):
            continue

        mags = np.array(data["mag"], dtype=float)
        mag_errs = np.array(data['mag_err'], dtype=float)
        mag_ul = np.array(data.get('mag_ul', np.full_like(mags, np.nan)), dtype=float)
        lim_mag = np.array(data.get('limiting_mag', np.full_like(mags, np.nan)), dtype=float)

        mags_corr = np.full_like(mags, np.nan)
        mag_errs_corr = np.full_like(mags, np.nan)
        mag_ul_corr = np.full_like(mag_ul, np.nan)
        lim_mag_corr = np.full_like(lim_mag, np.nan)

        # detections
        for i, m in enumerate(mags):
            if np.isfinite(m):
                m_corr, A_total = correct_extinction(m, filt, Av_host, Av_MW)
                mags_corr[i] = m_corr - dist_mod # intrinsic abs mag
                # Error propagation
                mag_errs_corr[i] = np.sqrt(mag_errs[i]**2 + sigma_A**2)

        # non-detections
        for i, m_ul in enumerate(mag_ul):
            if np.isfinite(m_ul):
                m_ul_corr, A_total = correct_extinction(m_ul, filt, Av_host, Av_MW)
                mag_ul_corr[i] = m_ul_corr - dist_mod
        for i, lm in enumerate(lim_mag):
            if np.isfinite(lm):
                lm_corr, A_total = correct_extinction(lm, filt, Av_host, Av_MW)
                lim_mag_corr[i] = lm_corr - dist_mod

        data["mag_intrinsic"] = mags_corr
        data["mag_intrinsic_err"] = mag_errs_corr
        data["mag_ul_intrinsic"] = mag_ul_corr
        data["limiting_mag_intrinsic"] = lim_mag_corr
        data["A_total"] = A_total

    # --- Convert to fluxes if requested ---
    if flux:
        forced = convert_ZTF_mag_mJy({"forced": forced}, forced=True)["forced"]

        # calculate intrinsic fluxes
        for filt, data in forced.items():
            A_total = data.get("A_total", 0.0)
            f_corr = 10**(0.4 * (A_total + dist_mod))

            for key in ["flux_mJy", "flux_err_mJy", "lim_flux_mJy"]:
                if key in data:
                    flux = np.array(data[key], dtype=float)
                    flux_intrinsic = flux * f_corr
                    data[f"{key}_intrinsic"] = flux_intrinsic
            
            # DN fallback
            for key in ["flux", "flux_err"]:
                if key in data:
                    flux = np.array(data[key], dtype=float)
                    flux_intrinsic = flux * f_corr
                    data[f"{key}_intrinsic"] = flux_intrinsic

    
    res["forced_corr"] = forced

    return res

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

###########################################################################
## ---------- WISE LIGHT CURVE DATA PARSING AND PLOTTING --------------- ##
###########################################################################

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
        filename = glob.glob(f"data/ztf_snii_lcs_WISE/lightcurve_{oid}_*.json")[0]
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
                     xlim=(None, None), ztf_flux=False, mode="stacked",
                     scale_wise=True, baseline_ref="ztf", baseline_dt=100,
                     ref_band="r", logy=False, savepath=None):   
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
            print("Converting ZTF mag to mJy...")
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
        if savepath:
            plt.savefig(savepath, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {savepath}")
        else:
            plt.show()

    elif mode == "overlay":
        # ---------------------------
        # Single-panel figure
        # ---------------------------
        fig, ax = plt.subplots(figsize=(10,6))

        # --- ZTF LC ---
        if ztf_flux:
            print("Converting ZTF mag to mJy...")
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

        ax.set_title(f"ZTF + WISE Light Curve: {oid}", fontsize=16)
        ax.set_xlabel("MJD", fontsize=14)
        ax.set_ylabel("Flux (mJy)", fontsize=14)
        
        handles, labels = ax.get_legend_handles_labels()
        # keep only ZTF detections and WISE bands
        keep = ["ZTF_g", "ZTF_r", "ZTF_i", "W1", "W2"]
        filtered = [(h, l) for h, l in zip(handles, labels) if l in keep]

        if filtered:
            handles, labels = zip(*filtered)
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1))

        ax.grid(True, alpha=0.4)

        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0, max(ax.get_ylim()))

        if savepath:
            plt.savefig(savepath, format="pdf", bbox_inches="tight")
            print(f"Saved plot to {savepath}")
        else:
            plt.show()

    else:
        raise ValueError("Invalid mode. Choose 'stacked' or 'overlay'.")