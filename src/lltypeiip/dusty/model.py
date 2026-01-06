import re
import numpy as np
from pathlib import Path
from astropy.io import ascii


class DustyModel:
    """
    Container for a single DUSTY model with its parameters and SED.
    
    Attributes
    ----------
    folder : Path
        Directory containing the model files.
    Tstar : float
        Stellar temperature (K).
    Tdust : float
        Inner dust temperature (K).
    tau : float
        Optical depth at tau_wavelength_microns.
    shell_thickness : float
        Shell thickness R_out / R_in.
    lam_um : np.ndarray
        Wavelength grid in micrometers.
    lamFlam : np.ndarray
        λF_λ flux in erg s⁻¹ cm⁻².
    scale : float or None
        Best-fit scale factor (filled after fitting).
    chi2 : float or None
        Chi-squared value (filled after fitting).
    dof : int or None
        Degrees of freedom (filled after fitting).
    chi2_red : float or None
        Reduced chi-squared (filled after fitting).
    x_plot : np.ndarray or None
        X-axis values for plotting (filled after fitting).
    y_scaled : np.ndarray or None
        Scaled flux values for plotting (filled after fitting).
    """
    
    def __init__(self, folder, Tstar, Tdust, tau, shell_thickness, lam_um, lamFlam):
        self.folder = Path(folder)
        self.Tstar = float(Tstar)
        self.Tdust = float(Tdust)
        self.tau = float(tau)
        self.shell_thickness = float(shell_thickness)
        self.lam_um = np.array(lam_um, float)
        self.lamFlam = np.array(lamFlam, float)

        # Filled after fitting
        self.scale = None
        self.chi2 = None
        self.dof = None
        self.chi2_red = None
        self.x_plot = None
        self.y_scaled = None
    
    def __repr__(self):
        return (f"DustyModel(Tstar={self.Tstar}K, Tdust={self.Tdust}K, "
                f"tau={self.tau}, thickness={self.shell_thickness})")


def load_dusty_grid(grid_dir):
    """
    Load DUSTY model grid from directory structure.
    
    Looks for subfolders named like 'Tstar_4000_Tdust_1100_tau_0_03_thick_2_0'
    containing 'sed.dat' with two columns: wavelength [μm], flux [λF_λ].
    """
    grid_dir = Path(grid_dir)
    
    # Match directory names like: Tstar_4000_Tdust_1100_tau_0_03_thick_2_0
    rx = re.compile(r"^Tstar_(\d+)_Tdust_(\d+)_tau_([0-9_]+)_thick_([0-9_]+)$")

    models = []
    for sub in sorted(grid_dir.iterdir()):
        if not sub.is_dir():
            continue
        match = rx.fullmatch(sub.name)
        if not match:
            continue
        sed_path = sub / "sed.dat"
        if not sed_path.exists():
            continue

        # Parse parameters from directory name
        tstar = float(match.group(1))
        tdust = float(match.group(2))
        tau = float(match.group(3).replace("_", "."))
        shell_thickness = float(match.group(4).replace("_", "."))

        # Load SED data
        try:
            table = ascii.read(sed_path, names=["lam", "flux"], comment="#", fast_reader=False)
            lam_um = np.array(table["lam"], float)
            lamFlam = np.array(table["flux"], float)
            
            models.append(DustyModel(sub, tstar, tdust, tau, shell_thickness, lam_um, lamFlam))
        except Exception as e:
            # Skip models with read errors
            continue

    if not models:
        raise RuntimeError(f"No DUSTY models with sed.dat found under {grid_dir}")
    
    return models
