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
    

    