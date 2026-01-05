import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo

def integrate_Fbol_from_lamFlam(lam_um, lamFlam):
    """
    Compute bolometric flux by integrating lamFlam over lam_um.
    """

    F_bol = np.trapezoid(lamFlam, np.log(lam_um))

    return F_bol

def luminosity_distance_cm(z):
    return cosmo.luminosity_distance(z).to(u.cm).value

def Fbol_to_Lbol(F_bol, z):
    """
    Convert bolometric flux to bolometric luminosity.
    """
    dl = luminosity_distance_cm(z)
    L_bol = 4 * np.pi * dl**2 * F_bol
    return L_bol

def MNi_from_tail(Lbol, t_days):
    """
    Estimate M_Ni from bolometric luminosity on the tail. Equation from Goldberg 2019.

    Lbol: bolometric luminosity in erg/s
    t_days: time since explosion in days
    could add something for not full trapping
    """

    tau_Co = 111.3  # cobalt decay time in days

    L0= 1.45e43 # erg/s per Msun of Ni-56

    MNi = (Lbol / L0)*np.exp(t_days / tau_Co)

    return MNi