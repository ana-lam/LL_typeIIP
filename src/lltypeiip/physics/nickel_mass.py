import numpy as np

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