import numpy as np
import pandas as pd
from pathlib import Path
from astropy import coordinates as coord
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo

from ..config import config, EXTINCTION_RV, EXTINCTION_SF11_SCALE, EXTINCTION_COEFFS, setup_dustmaps
from .ztf import convert_ZTF_mag_mJy


def _add_MW_extinction(meta_df, coords_df_path=None):
    """
    Add Milky Way extinction E(B-V) and A_V,MW to metadata DataFrame.
    Following Schlafly & Finkbeiner (2011) scaling (0.86 * SFD),
    Cardelli et al. (1989) law, and R_V = 3.1.
    """
    # Initialize dustmaps
    sfd = setup_dustmaps()
    R_V = EXTINCTION_RV
    SF11_SCALE = EXTINCTION_SF11_SCALE
    
    # ztf coords
    if coords_df_path is None:
        coords_df_path = Path(config.paths.ztf_coords)
    
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
    
    # Save to output file
    output_path = Path(config.paths.data_dir) / "zenodo_metasample_with_MW.csv"
    meta_df.to_csv(output_path, index=False)
    
    return meta_df

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

    # Band-dependent A_lambda / A_V ratios from config
    filt_key = f"ZTF_{filt.lower()[0]}"
    factor = EXTINCTION_COEFFS.get(filt_key, 1.0)  # fallback to 1.0 if unknown filter

    # total extinction in this band
    A_total = factor * (Av_MW + Av_host)

    mag_corr = mag - A_total
    return mag_corr, A_total

def intrinsic_lc_corrections(res, meta_df, sigma_A=True, flux=True):
    """
    Apply distance and extinction corrections to ZTF light curve data.
    
    Parameters
    ----------
    res : dict
        ZTF results dictionary with 'forced' photometry data.
    meta_df : pd.DataFrame
        Metadata dataframe with redshift, Avhost, A_V_MW columns.
    sigma_A : bool or float, optional
        If True, uses default 0.05 mag uncertainty. If False, no extra uncertainty.
        If float, uses that value. Default True.
    flux : bool, optional
        If True, also converts to flux units. Default True.
    
    Returns
    -------
    dict
        Updated results dictionary with 'forced_corr' key containing
        intrinsic (distance and extinction corrected) photometry.
    """
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