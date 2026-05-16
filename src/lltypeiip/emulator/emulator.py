import time
import pickle
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn


# -- helper functions
def _build_mlp(n_in, n_out, hidden=512, depth=5):
    """Build the MLP architecture. Used by both training and inference."""

    layers = [nn.Linear(n_in, hidden), nn.SiLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.SiLU()]
    layers += [nn.Linear(hidden, n_out)]
    return nn.Sequential(*layers)

class DustyNNEmulator:
    """
    A simple neural network emulator for DUSTY models. Drop-in replacement for the DUSTY runner 
    for use in MCMC.

    - Reads pre-built .npz files.
    - No DUSTY binary needed.
    - No cache files created.

    Usage:
        emu = DustyNNEmulator("dusty_runs/dusty_nn_emulator_bb_thick2_0.npz")
        lam_um, lamFlam, r1 = emu.evaluate_model(tstar=5000, tdust=400, tau=0.5)
    
    Build the emulator file with:
        python -m lltypeiip.inference.build_emulator
    """

    def __init__(self, emulator_path):
        
        emulator_path = Path(emulator_path)
        if not emulator_path.exists():
            raise FileNotFoundError("Emulator file not found.")
        
        self._backend = "pytorch"

        print(f"[DustyNNEmulator] Initialized {self._backend} with {emulator_path}")

        t0 = time.time()

        d = np.load(emulator_path, allow_pickle=True)

        self.lam_ref = d['lam_ref']
        self.X_mean = d['X_mean']
        self.X_std = d['X_std']
        self.Y_mean = d['Y_mean']
        self.Y_std = d['Y_std']
        self._n_lam = len(self.lam_ref)

        self._load_torch(d)

        if "X_train" in d.files:
            X_train = d['X_train']
            self._bounds = {
                "tstar": (X_train[:, 0].min(), X_train[:, 0].max()),
                "tdust": (X_train[:, 1].min(), X_train[:, 1].max()),
                "log10tau": (X_train[:, 2].min(), X_train[:, 2].max()),
            }
        else:
            self._bounds = None
        
        # per-parameter extrapolation warning flags
        self._warned = {"tstar": False, "tdust": False, "log10tau": False}

        print(f"[DustyNNEmulator] Ready in {time.time() - t0:.1f} seconds "
              f"| lambda points: {self._n_lam} "
              f"| backend: {self._backend}")
    
    # --- Helpers ---
    def _check_bounds(self, tstar, tdust, log10_tau):
        """Warn once if parameters are outside training range."""
        if self._bounds is None:
            return
        checks = [
            ("tstar",    tstar,    self._bounds["tstar"]),
            ("tdust",    tdust,    self._bounds["tdust"]),
            ("log10tau", log10_tau, self._bounds["log10tau"]),
        ]
        for name, val, (lo, hi) in checks:
            if not (lo <= val <= hi) and not self._warned[name]:
                print(
                    f"[DustyNNEmulator] WARNING: {name}={val:.3f} is outside "
                    f"training range [{lo:.3f}, {hi:.3f}]. "
                    f"Extrapolating — results may be unreliable."
                )
                self._warned[name] = True
    
    def __repr__(self):
        return (
            f"DustyNNEmulator("
            f"backend={self._backend!r}, "
            f"n_lam={self._n_lam}, "
            f"bounds={self._bounds})"
        )

    
    # ---- Loaders ----
    def _load_torch(self, d):
        prefix = "w__"
        w_keys = sorted([k for k in d.files if k.startswith(prefix)])

        if not w_keys:
            raise RuntimeError("No PyTorch weights found in the emulator file.")
        
        # reconstruct state dict - keys stored as "w__layer__weight"
        state = {
            k[len(prefix):].replace("__", "."): torch.tensor(d[k]) for k in w_keys
        }

        # infer model architecture from the state dict
        linear_keys = [k for k in state if k.endswith(".weight")]
        hidden_size = state[linear_keys[0]].shape[0]
        depth = len(linear_keys) - 1 # number of hidden layers

        self._model = _build_mlp(n_in=3, n_out=self._n_lam, 
                                    hidden=hidden_size, depth=depth
                                    )
        self._model.load_state_dict(state)
        self._model.eval()

        print(f"[DustyNNEmulator] Architecture: "
                f"3 -> [{hidden_size}]x{depth} -> {self._n_lam} "
                "(SiLU activations)")
        
    def evaluate_model(self, tstar, tdust, tau, shell_thickness=None,
                        **kwargs):
        """
        Evaluate the emulated DUSTY SED for given parameters.
        Replacement for DustyRunner.evaluate_model().

        Parameters
        ----------
        tstar : float
            Stellar temp [K]
        tdust : float
            Temperature of the dust [K]
        tau : float
            Optical depth (linear, not log)
        shell_thickness : float, optional
            Thickness of the dust shell.

        Returns
        -------
        lam_um : array
        lamFlam : array
        r1 : None
        """

        log10_tau = float(np.log10(max(float(tau), 1e-10))) # in case tau is 0 or negative
        x = np.array([[float(tstar), float(tdust), log10_tau]], dtype=np.float32)

        self._check_bounds(float(tstar), float(tdust), log10_tau)

        x_norm = ((x - self.X_mean) / self.X_std).astype(np.float32)

        with torch.no_grad():
            y_norm = self._model(torch.tensor(x_norm)).numpy()[0]
        
        log_sed = (y_norm * self.Y_std) + self.Y_mean
        lamFlam = np.exp(log_sed)

        return self.lam_ref, lamFlam, None

class DustyTemplateEmulator:
    """
    RegularGridInterpolator emulator for template (Spectrum=5) mode.
    Inputs : (T_dust, log10_tau, phase_days)
    Output : (lam_um, lamFlam)

    Much simpler than the NN emulator because the template cache is a grid.
    """

    def __init__(self, emulator_path):
        from scipy.interpolate import RegularGridInterpolator

        emulator_path = Path(emulator_path)
        if not emulator_path.exists():
            raise FileNotFoundError(f"Template emulator not found: {emulator_path}")

        print(f"[DustyTemplateEmulator] Loading from {emulator_path}...")
        t0 = time.time()

        d = np.load(emulator_path, allow_pickle=True)

        self.lam_ref   = d["lam_ref"].astype(np.float64)
        self.tdust_u   = d["tdust_u"].astype(np.float64)
        self.log10tau_u= d["log10tau_u"].astype(np.float64)
        self.phase_u   = d["phase_u"].astype(np.float64)
        self._n_lam    = len(self.lam_ref)

        sed_cube = d["sed_cube"].astype(np.float64)  # (tdust, tau, phase, lam)

        # Interpolate in log-SED space
        log_sed = np.log(np.where(sed_cube > 0, sed_cube, 1e-300))

        self._interp = RegularGridInterpolator(
            (self.tdust_u, self.log10tau_u, self.phase_u),
            log_sed,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        self._bounds = {
            "tdust":    (self.tdust_u[0],    self.tdust_u[-1]),
            "log10tau": (self.log10tau_u[0],  self.log10tau_u[-1]),
            "phase":    (self.phase_u[0],     self.phase_u[-1]),
        }
        self._warned = {"tdust": False, "log10tau": False, "phase": False}

        print(f"[DustyTemplateEmulator] Ready in {time.time()-t0:.1f}s  "
              f"| T_dust: {len(self.tdust_u)}  "
              f"| tau: {len(self.log10tau_u)}  "
              f"| phase: {len(self.phase_u)}  "
              f"| lambda: {self._n_lam}")
        
    def evaluate_model(self, tstar=None, tdust=None, tau=None,
                       shell_thickness=None, template=None,
                       phase_days=None, template_tag=None, **kwargs):
        """Drop-in for DustyRunner.evaluate_model() in template mode."""
        log10_tau  = float(np.log10(max(float(tau), 1e-10)))
        phase      = float(phase_days)

        # Clip to grid bounds
        tdust_c    = np.clip(float(tdust),  *self._bounds["tdust"])
        log10tau_c = np.clip(log10_tau,      *self._bounds["log10tau"])
        phase_c    = np.clip(phase,          *self._bounds["phase"])

        self._check_bounds(float(tdust), log10_tau, phase)

        point   = np.array([[tdust_c, log10tau_c, phase_c]])
        log_sed = self._interp(point)[0]
        lamFlam = np.exp(log_sed).astype(np.float64)

        return self.lam_ref, lamFlam, None
    
    def _check_bounds(self, tdust, log10_tau, phase):
        checks = [
            ("tdust",    tdust,    self._bounds["tdust"]),
            ("log10tau", log10_tau, self._bounds["log10tau"]),
            ("phase",    phase,    self._bounds["phase"]),
        ]
        for name, val, (lo, hi) in checks:
            if not (lo <= val <= hi) and not self._warned[name]:
                print(f"[DustyTemplateEmulator] WARNING: {name}={val:.3f} "
                      f"outside [{lo:.3f}, {hi:.3f}]. Extrapolating.")
                self._warned[name] = True

    def __repr__(self):
        return (f"DustyTemplateEmulator("
                f"tdust={self.tdust_u[[0,-1]]}, "
                f"log10tau={self.log10tau_u[[0,-1]]}, "
                f"phase={self.phase_u[[0,-1]]})")
    


