import re
import numpy as np
import pandas as pd
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

    # optional for template models
    oid: str or None
        Object identifier (for template grids).
    phase_days: float or None
        Phase in days (for template grids).
    template_tag: str or None
        Template tag (for template grids).
    
    # Filled after fitting
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
    
    def __init__(self, folder, Tstar, Tdust, tau, shell_thickness, lam_um, lamFlam,
                 oid=None, phase_days=None, template_tag=None, npz_path=None,
                 outpath=None):
        self.folder = Path(folder)
        self.Tstar = float(Tstar)
        self.Tdust = float(Tdust)
        self.tau = float(tau)
        self.shell_thickness = float(shell_thickness)
        self.lam_um = np.array(lam_um, float)
        self.lamFlam = np.array(lamFlam, float)

        # Optional template model attributes
        self.oid = oid
        self.phase_days = phase_days
        self.template_tag = template_tag

        # Filled after fitting
        self.scale = None
        self.chi2 = None
        self.dof = None
        self.chi2_red = None
        self.x_plot = None
        self.y_scaled = None

        self.npz_path = str(npz_path) if npz_path is not None else None
        self.outpath = str(outpath) if outpath is not None else None
    
    def __repr__(self):
        base = f"DustyModel(Tstar={self.Tstar}K, Tdust={self.Tdust}K, tau={self.tau}, thickness={self.shell_thickness}"
        if self.oid is not None:
            base += f", oid={self.oid}"
        if self.phase_days is not None:
            base += f", phase={self.phase_days:.1f}d"
        return base + ")"


def _make_folder_name(row, is_template_grid=False):
    """
    Create a fake folder path from CSV row for DustyModel compatibility.
    """
    tstar = int(row.get("tstar", row.get("tstar_dummy", 6000)))
    tdust = int(row["tdust"])
    tau = float(row["tau"])
    thick = float(row["shell_thickness"])
    
    if is_template_grid:
        oid = row.get("oid", "unknown")
        phase = row.get("phase_days", 0)
        folder_name = f"TEMPLATE_{oid}_phase{phase:.1f}_Td{tdust}_tau{tau:.3f}_thick{thick:.1f}"
    else:
        folder_name = f"Tstar_{tstar}_Tdust_{tdust}_tau_{tau}_thick_{thick}"
    
    folder_name = folder_name.replace(".", "_")
    return Path(folder_name)


def load_dusty_grid(grid_dir=None, csv_path=None, prefer_npz=True, 
                    oid=None, template_tag=None, top_k=None):
    """
    Load DUSTY model grid from NPZ cache, csv, or sed.dat files.

    First load from csv with npz_path.
    Second, try to load from outpath column (loads from sed.dat).
    Third, from directory structure (scan for sed.dat)
        Looks for subfolders named like 'Tstar_4000_Tdust_1100_tau_0_03_thick_2_0'
        containing 'sed.dat' with two columns: wavelength [μm], flux [λF_λ].
    """
    if csv_path is not None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path)

        is_template_grid = all(col in df.columns for col in ['oid', 'phase_days', 'template_tag'])

        # Filter by oid (template grids only)
        if is_template_grid and oid is not None:
            df = df[df["oid"].astype(str) == str(oid)].copy()
            if df.empty:
                raise RuntimeError(f"No rows found for oid={oid}")
        
        # Filter by template_tag (template grids only)
        if is_template_grid and template_tag is not None:
            df = df[df["template_tag"].astype(str) == str(template_tag)].copy()
            if df.empty:
                raise RuntimeError(f"No rows found for template_tag={template_tag}")
            
        # Filter successful models
        if "ierror" in df.columns:
            df_ok = df[df["ierror"] == 0].copy()
        else:
            df_ok = df.copy()

        # Sort by chi2_red if available
        if "chi2_red" in df_ok.columns:
            df_ok = df_ok.sort_values("chi2_red").reset_index(drop=True)

        # Limit to top K (template grids only)
        if top_k is not None:
            df_ok = df_ok.head(int(top_k))

        has_npz = "npz_path" in df_ok.columns and df_ok["npz_path"].notna().any()
        has_outpath = "outpath" in df_ok.columns and df_ok["outpath"].notna().any()

        if not (has_npz or has_outpath):
            raise RuntimeError("CSV has no npz_path or outpath columns with valid data")

        models = []
        load_failures = []

        for idx, row in df_ok.iterrows():
            loaded = False
            npz_path_str = None
            outpath_str = None
            
            if prefer_npz and has_npz and pd.notna(row.get("npz_path")):
                npz_path = Path(row["npz_path"])
                npz_path_str = str(npz_path)
                if npz_path.exists():
                    try:
                        z = np.load(npz_path, allow_pickle=False)
                        lam_um = z["lam_um"]
                        lamFlam = z["lamFlam"]
                        
                        # Create model with metadata
                        folder = _make_folder_name(row, is_template_grid)

                        if "outpath" in row and pd.notna(row["outpath"]):
                            outpath_str = str(row["outpath"])
                        
                        model = DustyModel(
                            folder=folder,
                            Tstar=row.get("tstar", row.get("tstar_dummy", 6000)),
                            Tdust=row["tdust"],
                            tau=row["tau"],
                            shell_thickness=row["shell_thickness"],
                            lam_um=lam_um,
                            lamFlam=lamFlam,
                            oid=row.get("oid") if is_template_grid else None,
                            phase_days=row.get("phase_days") if is_template_grid else None,
                            template_tag=row.get("template_tag") if is_template_grid else None,
                            npz_path=npz_path_str,
                            outpath=outpath_str 
                        )
                        
                        # Populate chi2 if already in CSV
                        if "scale" in row and pd.notna(row["scale"]):
                            model.scale = float(row["scale"])
                        if "chi2" in row and pd.notna(row["chi2"]):
                            model.chi2 = float(row["chi2"])
                        if "chi2_red" in row and pd.notna(row["chi2_red"]):
                            model.chi2_red = float(row["chi2_red"])
                        
                        models.append(model)
                        loaded = True
                    except Exception as e:
                        load_failures.append(f"NPZ {npz_path.name}: {e}")

            if not loaded and has_outpath and pd.notna(row.get("outpath")):
                sed_path = Path(row["outpath"])
                outpath_str = str(sed_path)
                if sed_path.exists():
                    try:
                        table = ascii.read(sed_path, names=["lam", "flux"], 
                                         comment="#", fast_reader=False)
                        lam_um = np.array(table["lam"], float)
                        lamFlam = np.array(table["flux"], float)
                        
                        folder = sed_path.parent
                        model = DustyModel(
                            folder=folder,
                            Tstar=row.get("tstar", row.get("tstar_dummy", 6000)),
                            Tdust=row["tdust"],
                            tau=row["tau"],
                            shell_thickness=row["shell_thickness"],
                            lam_um=lam_um,
                            lamFlam=lamFlam,
                            oid=row.get("oid") if is_template_grid else None,
                            phase_days=row.get("phase_days") if is_template_grid else None,
                            template_tag=row.get("template_tag") if is_template_grid else None,
                            npz_path=npz_path_str if npz_path_str else None,
                            outpath=outpath_str
                        )
                        models.append(model)
                        loaded = True
                    except Exception as e:
                        load_failures.append(f"sed.dat {sed_path.name}: {e}")
            
            if not loaded:
                load_failures.append(f"Row {idx}: no valid npz_path or outpath")
        
        if not models:
            error_summary = "\n".join(load_failures[:10])
            raise RuntimeError(
                f"No models could be loaded from CSV.\n"
                f"Checked {len(df_ok)} rows.\n"
                f"Sample errors:\n{error_summary}"
            )

        if load_failures and len(load_failures) < len(df_ok):
            print(f"Warning: Failed to load {len(load_failures)}/{len(df_ok)} models")
        
        return models
    
    if grid_dir is None:
        raise ValueError("Must provide either csv_path or grid_dir")
    
    grid_dir = Path(grid_dir)
    if not grid_dir.exists():
        raise FileNotFoundError(f"Grid directory not found: {grid_dir}")
    
    # Match directory names like: Tstar_4000_Tdust_1100_tau_0_03_thick_2_0
    rx = re.compile(r"^Tstar_(\d+)_Tdust_(\d+)_tau_([0-9_]+)_thick_([0-9_]+)$")

    models = []
    for sub in sorted(grid_dir.iterdir()):
        if not sub.is_dir():
            continue
        match = rx.fullmatch(sub.name)
        if not match:
            continue
        
        # Parse parameters from directory name
        tstar = float(match.group(1))
        tdust = float(match.group(2))
        tau = float(match.group(3).replace("_", "."))
        shell_thickness = float(match.group(4).replace("_", "."))
        
        # Load from sed.dat
        sed_path = sub / "sed.dat"
        if sed_path.exists():
            try:
                table = ascii.read(sed_path, names=["lam", "flux"], 
                                 comment="#", fast_reader=False)
                lam_um = np.array(table["lam"], float)
                lamFlam = np.array(table["flux"], float)
                
                models.append(DustyModel(sub, tstar, tdust, tau, 
                                        shell_thickness, lam_um, lamFlam))
            except Exception:
                continue

    if not models:
        raise RuntimeError(
            f"No DUSTY models found under {grid_dir}.\n"
            f"Hint: If you have a CSV file, use csv_path argument for faster loading:\n"
            f"  load_dusty_grid(csv_path='path/to/model_grid_summary.csv')"
        )

    return models
