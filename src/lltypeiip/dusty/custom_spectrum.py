import numpy as np

class NugentIIPSeries:
    def __init__(self, template_path):
        """
        Load the Nugent Type IIP SN spectral series from a file.
        """

        arr = np.loadtxt(template_path)
        self.times = np.unique(arr[:, 0])
        self.lam_A = np.unique(arr[:, 1])

        nt = self.times.size
        nl = self.lam_A.size

        nrows = arr.shape[0]
        if nrows != nt * nl:
            raise ValueError(
                f"Template isn't a full grid: nrows={nrows}, nt*nl={nt*nl}."
            )

        if not np.allclose(arr[:nl, 0], self.times[0]):
            raise ValueError(
                "Template rows not grouped by time (time-major ordering expected). "
                "Need a group-by-time build instead of reshape."
            )

        # reshape fluxes into (nt, nl) for time x wavelength
        self.f_grid = arr[:, 2].reshape((nt, nl))
    
    @classmethod
    def from_single_spectrum(cls, flm_path, z=0.0, time_lo=-9999.0, time_hi=9999.0):
        """
        Build a phase-independent 'series' directly from a single flux-calibrated
        .flm spectrum (wl(A), flm, [err])

        The spectrum is de-redshifted to rest frame and duplicated across two
        bracketing time values (time_lo, time_hi), so spectrum_at(phase_days)
        returns this same spectrum for any requested phase -- either by clamping
        (phase outside [time_lo, time_hi]) or by interpolating between two
        identical arrays (phase inside the range).

        Parameters
        ----------
        flm_path : str or Path
            Path to the .flm spectrum file.
        z : float
            Redshift used to de-redshift wl -> rest frame.
        time_lo, time_hi : float
            Bracketing phase values written internally; pick a range wide
            enough to cover every phase_days you'll ever query.
        """
        obj = cls.__new__(cls)

        data = np.loadtxt(flm_path, comments="#")
        wl_obs = data[:, 0]
        flux = data[:, 1] * 1e-15  # -> erg/s/cm^2/A (de Jaeger et al. 2019 convention)

        wl_rest = wl_obs / (1 + z)

        order = np.argsort(wl_rest)
        wl_rest = wl_rest[order]
        flux = flux[order]
        wl_rest, unique_idx = np.unique(wl_rest, return_index=True)
        flux = flux[unique_idx]

        obj.times = np.array([time_lo, time_hi])
        obj.lam_A = wl_rest
        obj.f_grid = np.vstack([flux, flux])  # identical spectrum at both times

        return obj


    @property
    def lam_um(self):
        return self.lam_A * 1e-4  # convert Angstroms to microns
    
    def spectrum_at(self, phase_days, extrapolate="clamp", floor=1e-40):
        """
        Get the spectrum at a given phase (in days).
        Log-linear interpolation is used between phases to keep flux positive.
        """

        t = float(phase_days)
        ts = self.times

        if t <= ts[0]:
            # return earliest spectrum if before first time
            if extrapolate == "clamp":
                return self.lam_um, self.f_grid[0]
            j0, j1 = 0, 1
        elif t >= ts[-1]:
            # return latest spectrum if after last time
            if extrapolate == "clamp":
                return self.lam_um, self.f_grid[-1]
            j0, j1 = -2, -1
        else:
            j1 = np.searchsorted(ts, t)
            j0 = j1 - 1
        
        t0, t1 = ts[j0], ts[j1]
        f0, f1 = self.f_grid[j0], self.f_grid[j1]
        # linear interpolation weight
        w = (t - t0) / (t1 - t0)

        # log-linear interpolation of fluxes
        f0p = np.maximum(f0, floor)
        f1p = np.maximum(f1, floor)

        logf = (1 - w) * np.log(f0p) + w * np.log(f1p)

        return self.lam_um, np.exp(logf)
    

    