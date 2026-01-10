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
import tempfile
from collections import OrderedDict

from ..config import config

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

    def __init__(self, base_workdir=config.dusty.workdir, dusty_file_dir=config.dusty.dusty_file_dir,
                 dust_type="silicate", shell_thickness=2.0,
                 tstarmin=2000., tstarmax=12000., custom_grain_distribution=False,
                 tau_wavelength_microns=0.55, blackbody=True, logger=None, quiet=True,
                 cache_max=5000, cache_dir=None, cache_ndigits=4, use_tmp=True):
        
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

        self._cache = OrderedDict() # (tstar, tdust, tau, shell_thickness) -> (lam_um, lamFlam, r1)
        self.cache_max = int(cache_max)
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.cache_ndigits = int(cache_ndigits)
        self.use_tmp = bool(use_tmp)


        self.logger = logger
        self.quiet = quiet

    def _canonical_key(self, tstar, tdust, tau, shell_thickness):
            """Quantize parameters so cache hits happen."""
            tstar_i = int(round(float(tstar)))
            tdust_i = int(round(float(tdust)))
            tau_f = round(float(tau), self.cache_ndigits)
            thick_f = round(float(shell_thickness), self.cache_ndigits)
            return (tstar_i, tdust_i, tau_f, thick_f)

    @staticmethod
    def _leaf_from_ckey(ckey):
        tstar_i, tdust_i, tau_f, thick_f = ckey
        leaf = f"Tstar_{tstar_i}_Tdust_{tdust_i}_tau_{tau_f}_thick_{thick_f}".replace(".", "_")
        return leaf
    
    def _cache_get(self, ckey):
        if ckey in self._cache:
            self._cache.move_to_end(ckey)
            return self._cache[ckey]
        return None

    def _cache_set(self, ckey, value):
        self._cache[ckey] = value
        self._cache.move_to_end(ckey)
        if len(self._cache) > self.cache_max:
            self._cache.popitem(last=False)

    def _disk_cache_path(self, ckey):
        if self.cache_dir is None:
            return None
        leaf = self._leaf_from_ckey(ckey)
        return self.cache_dir / f"{leaf}.npz"
    
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

        ckey = self._canonical_key(tstar, tdust, tau, shell_thickness)

        # check in-memory cache
        hit = self._cache_get(ckey)
        if hit is not None:
            return hit

        # check disk cache
        dpath = self._disk_cache_path(ckey)
        if dpath is not None and dpath.exists():
            try:
                z = np.load(dpath, allow_pickle=False)
                lam_um = z["lam_um"]
                lamFlam = z["lamFlam"]
                r1 = float(z["r1"])
                out = (lam_um, lamFlam, r1)
                self._cache_set(ckey, out)
                return out
            except Exception:
                # corrupted cache file -> remove
                try:
                    dpath.unlink()
                except Exception:
                    pass
        
        pid_root = self.base_workdir / f"pid{os.getpid()}"
        pid_root.mkdir(parents=True, exist_ok=True)

        if self.use_tmp:
            tmp_cm = tempfile.TemporaryDirectory(dir=str(pid_root), prefix="dusty_")
            run_dir = Path(tmp_cm.name)
        else:
            # fallback: one dir per canonical model (still bounded if cache hits)
            run_dir = pid_root / self._leaf_from_ckey(ckey)
            run_dir.mkdir(parents=True, exist_ok=True)
            tmp_cm = None

        prev_cwd = os.getcwd()
        try:
            dusty_params = self._build_parameters(*ckey)

            runner = Dusty(
                parameters=dusty_params,
                dusty_working_directory=str(run_dir),
                dusty_file_directory=self.dusty_file_dir,
            )

            os.chdir(str(run_dir))
            with silence_fds():
                runner.generate_input()
                runner.run()
                lam, flx, npt, r1, ierror = runner.get_results()

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error running DUSTY in {run_dir}: {e}")
            out = (None, None, None)
            self._cache_set(ckey, out)
            return out

        finally:
            os.chdir(prev_cwd)
            if tmp_cm is not None:
                tmp_cm.cleanup()

        if ierror != 0:
            if self.logger:
                self.logger.warning(f"DUSTY ierror={ierror} for {ckey}")
            out = (None, None, None)
            self._cache_set(ckey, out)
            return out

        lam_um = np.asarray(lam, float)
        lamFlam = np.asarray(flx, float)
        r1 = float(r1)

        out = (lam_um, lamFlam, r1)

        # save disk cache (optional)
        if dpath is not None:
            try:
                np.savez_compressed(dpath, lam_um=lam_um, lamFlam=lamFlam, r1=r1)
            except Exception:
                pass

        self._cache_set(ckey, out)
        
        return out