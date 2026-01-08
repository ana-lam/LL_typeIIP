import numpy as np

def MNi_from_tail(Lbol, t_days):
    """
    Estimate M_Ni from bolometric luminosity on the tail. Equation from Goldberg 2019.

    Lbol: bolometric luminosity in erg/s
    t_days: time since explosion in days
    could add something for not full trapping
    """

    tau_Co = 111.3  # cobalt decay time in days, 56Co->56Fe
    tau_Ni = 8.8 # nickel decay time in days, 56Ni->56Co

    a_Co= 1.45e43 # erg/s per Msun of 56Co
    a_Ni = 6.45e43  # erg/s per Msun of 56Ni

    Q = (a_Ni * np.exp(-t_days / tau_Ni)) + (a_Co * np.exp(-t_days / tau_Co))

    MNi = Lbol / Q # in Msun

    return MNi