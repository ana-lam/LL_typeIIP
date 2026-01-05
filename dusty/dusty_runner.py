import os
import io
import contextlib
import shutil
import stat
from pathlib import Path
import numpy as np
from pydusty.dusty import Dusty, DustyParameters
from pydusty.parameters import Parameter
from astropy.io import ascii

@contextlib.contextmanager
def silence_fds():
    """
    Temporarily redirect low-level file descriptors 1 (stdout) and 2 (stderr)
    to /dev/null. This catches output from C/Fortran/child processes that
    bypass Python's sys.stdout/sys.stderr.
    """
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        os.dup2(devnull_fd, 1)
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(devnull_fd)
        os.close(saved_stdout)
        os.close(saved_stderr)

class DustyRunner:
    """
    Wrapper to run DUSTY models for specific parameters.
    """

    def __init__(self, base_workdir, dusty_file_dir,
                 dust_type="silicate", shell_thickness=2.0,
                 tstarmin=2000., tstarmax=12000., custom_grain_distribution=False,
                 tau_wavelength_microns=0.55, blackbody=True, logger=None, quiet=True):
        self.base_workdir = Path(base_workdir).resolve()
        self.base_workdir.mkdir(parents=True, exist_ok=True)

        self.dusty_file_dir = Path(dusty_file_dir).resolve()
        self.dusty_bin = self.dusty_file_dir / "dusty"
        if not self.dusty_bin.exists():
            raise FileNotFoundError(f"DUSTY binary not found at {self.dusty_bin}")

        self.shell_thickness_fixed = float(shell_thickness)
        self.dust_type_str = str(dust_type)
        self.tstarmin_val = float(tstarmin)
        self.tstarmax_val = float(tstarmax)
        self.custom_grain_distribution_val = bool(custom_grain_distribution)
        self.tau_wavelength_microns_val = float(tau_wavelength_microns)
        self.blackbody_val = bool(blackbody)

        self._cache = {}   # (tstar, tdust, tau, shell_thickness) -> (lam_um, lamFlam, r1)
        self.logger = logger
        self.quiet = quiet

    @staticmethod
    def _make_leaf(tstar, tdust, tau, shell_thickness, ndigits=6):
        """Canonical naming + canonical float key."""
        tstar_i = int(round(float(tstar)))
        tdust_i = int(round(float(tdust)))
        tau_f = round(float(tau), ndigits)
        thick_f = round(float(shell_thickness), ndigits)

        leaf = (f"Tstar_{tstar_i}_Tdust_{tdust_i}_"
                f"tau_{tau_f:.{ndigits}g}_thick_{thick_f:.{ndigits}g}").replace('.', '_')
        
        return leaf
    
    def _build_parameters(self, tstar, tdust, tau, shell_thickness):
        
        p_tstar = Parameter(name="tstar", value=float(tstar), is_variable=False)
        p_tdust = Parameter(name="tdust", value=float(tdust), is_variable=False)
        p_tau   = Parameter(name="tau",   value=float(tau),   is_variable=False)

        p_blackbody = Parameter(name="blackbody", value=self.blackbody_val)
        p_shell_thickness = Parameter(name="shell_thickness", value=float(shell_thickness))

        p_dust_type = Parameter(name="dust_type", value=self.dust_type_str)

        p_tstarmin = Parameter(name="tstarmin", value=self.tstarmin_val)
        p_tstarmax = Parameter(name="tstarmax", value=self.tstarmax_val)

        p_custom_grain_distribution = Parameter(
            name="custom_grain_distribution",
            value=self.custom_grain_distribution_val
        )
        p_tau_wav_micron = Parameter(
            name="tau_wav", value=self.tau_wavelength_microns_val, is_variable=False
        )

        dusty_params = DustyParameters(
            tstar=p_tstar,
            tdust=p_tdust,
            tau=p_tau,
            blackbody=p_blackbody,
            shell_thickness=p_shell_thickness,
            dust_type=p_dust_type,
            tstarmin=p_tstarmin,
            tstarmax=p_tstarmax,
            custom_grain_distribution=p_custom_grain_distribution,
            tau_wavelength_microns=p_tau_wav_micron,
        )
        return dusty_params
    
    def evaluate_model(self, tstar, tdust, tau, shell_thickness=None):
        """
        Run DUSTY (or load from cache/disk) and return (lam_um, lamFlam, r1).

        theta: (tstar, tdust, tau[, shell_thickness])
        """
        if shell_thickness is None:
            shell_thickness = self.shell_thickness_fixed

        key = (float(tstar), float(tdust), float(tau), float(shell_thickness))
        if key in self._cache:
            return self._cache[key]
        
        leaf = self._make_leaf(*key)
        run_dir = (self.base_workdir / leaf).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        sed_path = run_dir / "sed.dat"

        # do not run if already exists
        if sed_path.exists():
            table = ascii.read(sed_path, names=["lam", "flux"], comment="#", fast_reader=False)
            lam_um = np.array(table["lam"], float)
            lamFlam = np.array(table["flux"], float)

            try:
                with sed_path.open() as f:
                    first = f.readline().strip()
                # "# r1value"
                if first.startswith("#"):
                    r1 = float(first[1:].strip())
                else:
                    r1 = np.nan
            except Exception:
                r1 = np.nan

            self._cache[key] = (lam_um, lamFlam, r1)
            return lam_um, lamFlam, r1
        
        # run DUSTY
        # Silence all OS-level stdout/stderr from DUSTY & friends
        with silence_fds():
            dusty_params = self._build_parameters(*key)
            runner = Dusty(
                parameters=dusty_params,
                dusty_working_directory=str(run_dir),
                dusty_file_directory=self.dusty_file_dir,
            )

        # ensure DUSTY binary is present & executable
        dst = run_dir / "dusty"
        if not dst.exists():
            shutil.copy2(self.dusty_bin, dst)
            dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        
        if not os.access(dst, os.X_OK):
            raise PermissionError(f"DUSTY binary at {dst} is not executable.")
        
        # run DUSTY
        prev_cwd = os.getcwd()
        try:
            os.chdir(str(run_dir))
            os.environ["PATH"] = f"{str(run_dir)}:{str(Path(self.dusty_file_dir).resolve())}:{os.environ.get('PATH','')}"

            runner.generate_input()
            runner.run()
            lam, flx, npt, r1, ierror = runner.get_results()

        except Exception as e:
            os.chdir(prev_cwd)
            if self.logger:
                self.logger.error(f"Error running DUSTY in {run_dir}: {e}")
            self._cache[key] = (None, None, None)
            return None, None, None
        
        os.chdir(prev_cwd)

        if ierror != 0:
            if self.logger:
                self.logger.warning(f"DUSTY returned ierror={ierror} for {key}")
            self._cache[key] = (None, None, None)
            return None, None, None

        if ierror == 0:
            with sed_path.open("w") as f:
                f.write(f"# {r1}\n")
                f.write("lam, flux\n")
                for ind in range(len(lam)):
                    f.write(f"{lam[ind]}, {flx[ind]}\n")
        self._cache[key] = (lam, flx, r1)
        
        return lam, flx, r1