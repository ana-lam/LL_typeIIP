import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

def integrate_Fbol_from_lamFlam(lam_um, lamFlam, scale=1.0):
    """
    Compute bolometric flux by integrating lamFlam over lam_um.
    """

    lamFlam_scaled = lamFlam * scale
    lam_cm = lam_um * 1e-4
    
    F_bol = np.trapz(lamFlam_scaled, np.log(lam_cm))

    return F_bol #<- same as a, honestly redundant function integral(Flam dlam)=1

def luminosity_distance_cm(z):
    return cosmo.luminosity_distance(z).to(u.cm).value

def Fbol_to_Lbol(F_bol, z):
    """
    Convert bolometric flux to bolometric luminosity.
    """
    dl = luminosity_distance_cm(z)
    L_bol = 4 * np.pi * dl**2 * F_bol
    return L_bol