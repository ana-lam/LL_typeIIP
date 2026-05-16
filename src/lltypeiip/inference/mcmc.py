import os
from multiprocessing import get_context

import emcee
import numpy as np
from astropy import constants as const
from pathlib import Path

from ..dusty.runner import DustyRunner
from ..dusty.scaling import compute_chi2
from ..dusty.custom_spectrum import NugentIIPSeries
from ..sed.build import _prepare_sed_xy, _unwrap_sed
from .priors import log_prior
from ..emulator import DustyNNEmulator, DustyTemplateEmulator

def _ls_scale_and_chi2(lam_um, lamFlam, sed, a=None, y_mode="Flam", use_weights=True,
                       err_floor_frac=0.0, equal_weights=False):
    """Analytic best-fit scale and chi2 on the SED grid; if ``a`` is provided, only chi2.

    - If a is None, compute the analytic best-fit scale using detections only.
    - If a is provided from theta (MCMC), use it directly to compute chi2.
    """
    x_sed, y_sed, ey_sed, _, _ = _prepare_sed_xy(sed, y_mode=y_mode)
    is_ul = np.array(sed.get("is_ul", np.zeros(len(x_sed), dtype=bool)))

    if y_mode == "Flam":
        x_mod = lam_um
        y_mod = lamFlam
    elif y_mode == "Fnu":
        lam_cm = lam_um * 1e-4
        nu_mod = const.c.cgs.value / lam_cm
        Fnu_mod = lamFlam * (lam_cm / const.c.cgs.value)
        order = np.argsort(nu_mod)
        x_mod = nu_mod[order]
        y_mod = Fnu_mod[order]
    else:
        raise ValueError("y_mode must be 'Flam' or 'Fnu'")

    mask_data = (x_sed >= x_mod.min()) & (x_sed <= x_mod.max())

    det_mask = mask_data & ~is_ul & np.isfinite(ey_sed) & (ey_sed > 0)
    ul_mask  = mask_data & is_ul

    if not np.any(det_mask):
        return np.nan, np.inf

    x_det = x_sed[det_mask]
    y_det = y_sed[det_mask]
    ey_det = ey_sed[det_mask]

    if use_weights and err_floor_frac > 0.0:
        if equal_weights:
            ey_det = err_floor_frac * y_det
        else:
            ey_det = np.maximum(ey_det, err_floor_frac * y_det)

    y_mod_det = np.interp(x_det, x_mod, y_mod)

    # -- get analytic scale from detections only --
    if a is None:
        if use_weights and np.any(ey_det > 0):
            weights = 1.0 / np.clip(ey_det, 1e-99, np.inf) ** 2
            a = np.sum(weights * y_det * y_mod_det) / np.sum(weights * y_mod_det ** 2)
            # chi2 = np.sum(weights * (y_det - a * y_mod_det) ** 2)
        else:
            a = np.sum(y_det * y_mod_det) / np.sum(y_mod_det ** 2)
            # chi2 = np.sum((y_det - a * y_mod_det) ** 2)
    # -- compute chi2 --
    if use_weights:
        weights = 1.0 / np.clip(ey_det, 1e-99, np.inf) ** 2
        chi2 = np.sum(weights * (y_det - a * y_mod_det) ** 2)
    else:
        chi2 = np.sum((y_det - a * y_mod_det) ** 2)

    # -- UL penalities --
    if np.any(ul_mask):
        x_ul = x_sed[ul_mask]
        y_ul = y_sed[ul_mask]
        y_mod_ul = a * np.interp(x_ul, x_mod, y_mod)
        ul_sigma  = y_ul / 3.0
        excess = y_mod_ul - y_ul
        chi2 += np.sum(np.where(excess > 0, (excess / ul_sigma) ** 2, 0.0))

    return float(a), float(chi2)

##############################################
## ------- Likelihood / posterior --------- ##
##############################################

def log_likelihood(theta, sed, dusty_runner,
                   y_mode="Flam", use_weights=True,
                   template=None, template_tag="nugent_iip",
                   tstar_dummy=6000.0, shell_thickness=None,
                   sample_log10a=True, err_floor_frac=0.0, equal_weights=False):
    """
    sample_log10a=True: theta includes log10_a as a parameter to sample
    sample_log10a=False: theta has no log10_a; a solved analytically

    Template mode theta (3D): (tdust, log10_tau, log10_a)
    Blackbody mode theta (4D): (tstar, tdust, log10_tau, log10_a)
    """

    sed = _unwrap_sed(sed)

    theta = np.asarray(theta, float)

    a = None # analytic by default

    if sample_log10a:
        if theta.size == 3: # template
            tdust, log10_tau, log10_a = theta
            a = 10.0**log10_a
            tau = 10.0**log10_tau
            phase = sed.get("phase_days", sed.get("phase", None))
            if phase is None or not np.isfinite(float(phase)):
                return -np.inf
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                tstar=float(tstar_dummy),
                tdust=float(tdust),
                tau=float(tau),
                shell_thickness=shell_thickness,
                template=template,
                phase_days=float(phase),
                template_tag=str(template_tag),
            )
        elif theta.size == 4: # blackbody
            tstar, tdust, log10_tau, log10_a = theta
            a = 10.0**log10_a
            tau = 10.0**log10_tau
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                float(tstar),
                float(tdust), 
                float(tau), 
                shell_thickness=shell_thickness)
        else:
            return -np.inf
    else:
        if theta.size == 2: # template
            tdust, log10_tau = theta
            tau = 10.0**log10_tau
            phase = sed.get("phase_days", sed.get("phase", None))
            if phase is None or not np.isfinite(float(phase)):
                return -np.inf
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                tstar=float(tstar_dummy),
                tdust=float(tdust),
                tau=float(tau),
                shell_thickness=shell_thickness,
                template=template,
                phase_days=float(phase),
                template_tag=str(template_tag),
            )
        elif theta.size == 3: # blackbody
            tstar, tdust, log10_tau = theta
            tau = 10.0**log10_tau
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                float(tstar),
                float(tdust), 
                float(tau), 
                shell_thickness=shell_thickness)
        else:
            return -np.inf


    if lam_um is None or lamFlam is None:
        return -np.inf

    # a=None triggers analytic scale and chi2 computation in _ls_scale_and_chi2
    scale, chi2 = _ls_scale_and_chi2(
        lam_um, lamFlam, sed, a=a, y_mode=y_mode, use_weights=use_weights, err_floor_frac=err_floor_frac, equal_weights=equal_weights
    )
    if not np.isfinite(chi2):
        return -np.inf

    return -0.5 * chi2


def log_probability(theta, sed, dusty_runner, prior_config,
                    y_mode="Flam", use_weights=True,
                    template=None, template_tag="nugent_iip",
                    tstar_dummy=6000.0, shell_thickness=None,
                    sample_log10a=True, err_floor_frac=0.0, equal_weights=False):
    lp = log_prior(theta, **prior_config)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, sed, dusty_runner,
                        y_mode=y_mode, use_weights=use_weights, 
                        err_floor_frac=err_floor_frac, 
                        equal_weights=equal_weights,
                        template=template, template_tag=template_tag,
                        tstar_dummy=tstar_dummy, shell_thickness=shell_thickness,
                        sample_log10a=sample_log10a)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

##############################################
## --- Multiprocessing + Initialization --- ##
##############################################

def _pick_mp_context(prefer="spawn"):
    """Pick multiprocessing context.
    'fork' for Linux.
    """
    if prefer is None:
        prefer = "spawn"
    if prefer == "fork" and os.name != "nt":
        return get_context("fork")
    return get_context("spawn")

def _initialize_walkers(nwalkers, best, prior_config, ndim, init_mode="hybrid",
                        init_scales=None, max_tries=5000, random_seed=42):
    """
    init_mode:
        - "around_best": gaussian around best
        - "broad": uniform in prior bounds
        - "hybrid": gaussian around best with some fraction uniform in prior bounds
    init_scales keys:
        tstar_frac, tdust_frac (fractional around best)
        log10_tau, log10_a (dex)
    """

    rng = np.random.default_rng(random_seed)

    if init_scales is None:
        init_scales = dict(tstar_frac=0.15, 
                           tdust_frac=0.15, 
                           log10_tau=0.5, 
                           log10_a=0.5)

    tstar_bounds = prior_config.get("tstar_bounds", (1000.0, 10000.0))
    tdust_bounds = prior_config.get("tdust_bounds", (100.0, 1500.0))
    log10_tau_bounds = prior_config.get("log10_tau_bounds", (-4.0, 2.0))
    log10_a_bounds = prior_config.get("log10_a_bounds", (-20.0, 0.0))

    def draw_around():
        if ndim == 2: # template, analytic a
            return np.array([
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" not in best: # template, sample log10_a
            return np.array([
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" in best: # blackbody, analytic a
            return np.array([
                best["tstar"] * (1.0 + init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        else: # ndim==4, blackbody, sample log10_a
            return np.array([
                best["tstar"] * (1.0 + init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1.0 + init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)

    def draw_broad():
        """
        Use a 3-sigma envelope around best to keep walkers in reasonable territory.
        """
        scale = 3.0 # how many init_scales wide to draw from
        if ndim == 2: # template, analytic a
            cand = np.array([
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" not in best: # template, sample log10_a
            cand = np.array([
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + scale * init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        elif ndim == 3 and "tstar" in best: # blackbody, analytic a
            cand = np.array([
                best["tstar"] * (1. + scale * init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
            ], dtype=float)
        else: # ndim==4, blackbody, sample log10_a
            cand = np.array([
                best["tstar"] * (1. + scale * init_scales["tstar_frac"] * rng.standard_normal()),
                best["tdust"] * (1. + scale * init_scales["tdust_frac"] * rng.standard_normal()),
                best["log10_tau"] + scale * init_scales["log10_tau"] * rng.standard_normal(),
                best["log10_a"] + scale * init_scales["log10_a"] * rng.standard_normal(),
            ], dtype=float)
        # Clip to prior bounds
        if ndim == 2:
            cand[0] = np.clip(cand[0], *tdust_bounds)
            cand[1] = np.clip(cand[1], *log10_tau_bounds)
        elif ndim == 3 and "tstar" not in best:
            cand[0] = np.clip(cand[0], *tdust_bounds)
            cand[1] = np.clip(cand[1], *log10_tau_bounds)
            cand[2] = np.clip(cand[2], *log10_a_bounds)
        elif ndim == 3 and "tstar" in best:
            cand[0] = np.clip(cand[0], *tstar_bounds)
            cand[1] = np.clip(cand[1], *tdust_bounds)
            cand[2] = np.clip(cand[2], *log10_tau_bounds)
        else:
            cand[0] = np.clip(cand[0], *tstar_bounds)
            cand[1] = np.clip(cand[1], *tdust_bounds)
            cand[2] = np.clip(cand[2], *log10_tau_bounds)
            cand[3] = np.clip(cand[3], *log10_a_bounds)
        return cand
    
    # 80/20 split around best
    n_around = int(0.80 * nwalkers)
    
    p0 = np.zeros((nwalkers, ndim), dtype=float)

    for k in range(nwalkers):
        tries = 0
        while True:
            if init_mode == "around_best":
                cand = draw_around()
            elif init_mode == "broad":
                cand = draw_broad()
            elif init_mode == "hybrid":
                cand = draw_around() if (k < n_around) else draw_broad()
            else:
                raise ValueError("init_mode must be one of: 'around_best', 'broad', 'hybrid'.")

            if np.isfinite(log_prior(cand, **prior_config)):
                p0[k] = cand
                break

            tries += 1
            if tries >= max_tries:
                raise RuntimeError(
                        f"Could not find valid initial position for walker {k}. Try broader bounds or looser priors."
                    )
    
    return p0

##############################################
## -------------- MCMC driver ------------- ##
##############################################

def run_mcmc_for_sed(sed, grid_df, dusty_file_dir, workdir,
                     dust_type="silicate", shell_thickness=2.0,
                     template_path=None, template_tag="nugent_iip",
                     tstar_dummy=6000.0,
                     nwalkers=32, nsteps=4000, burn_in=1000, y_mode="Flam",
                     use_weights=True, n_cores=4, mp_prefer="fork",
                     random_seed=42, posterior_mode="data", mix_weight=0.3,
                     tdust_sigma_frac=0.2, log10_tau_sigma=0.3, log10_a_sigma=0.5,
                     init_mode="hybrid", init_scales=None, progress_every=None,
                     cache_dir=None, cache_ndigits=4, cache_max=5000, use_tmp=True,
                     run_tag=None, evaluator_mode="dusty", emulator_path=None,
                     sample_log10a=True, log10_a_range=3.0, err_floor_frac=0.0, 
                     equal_weights=False):
    """
    Run emcee for a given SED using DUSTY models.

    Always samples a (via log10_a).
    Use posterior_mode:
      - "data": bounds-only posterior
      - "anchored": regularized around grid best
      - "mixture": soft hint from grid + broad exploration

    Template mode:
      - needs sed['phase_days']
      - theta = (tdust, log10_tau, log10_a)  [ndim=3]
    """

    sed = _unwrap_sed(sed)

    if cache_dir is None and template_path is not None:
        cache_dir = str(Path(workdir) / "dusty_npz_cache_template")
    elif cache_dir is None:
        cache_dir = str(Path(workdir) / "dusty_npz_cache_blackbody")

    template = None
    template_mode = (template_path is not None)
    if template_mode:
        template = NugentIIPSeries(str(Path(template_path).resolve()))

    # Initialize the emulator if specified
    if evaluator_mode == "emulator":
        if emulator_path is None:
            raise ValueError("emulator_path must be specified when using emulator mode.")
        if not Path(emulator_path).exists():
            raise FileNotFoundError(f"Emulator file not found: {emulator_path}")
        if template_mode:
            dusty_runner = DustyTemplateEmulator(emulator_path)
        else:
            dusty_runner = DustyNNEmulator(emulator_path)

        print(f"[evaluator] mode=EMULATOR loaded from {emulator_path}")


    elif evaluator_mode == "dusty":
        print(f"[evaluator] mode=DUSTY cache={cache_dir}")

        dusty_runner = DustyRunner(
                            base_workdir=workdir,
                            dusty_file_dir=dusty_file_dir,
                            dust_type=dust_type,
                            shell_thickness=shell_thickness,
                            cache_dir=cache_dir,
                            cache_ndigits=cache_ndigits,
                            cache_max=cache_max,
                            use_tmp=use_tmp,
                            run_tag=run_tag
                        )
    else:
        raise ValueError(f"evaluator_mode must be either 'emulator' or 'dusty'")
    
    # best-fit from grid
    grid_df = grid_df.sort_values("chi2_red").reset_index(drop=True)
    best_row = grid_df.iloc[0]

    if template_mode:
        ndim = 3 if sample_log10a else 2
        best = {
            "tdust": float(best_row["tdust"]),
            "log10_tau": np.log10(float(best_row["tau"])),
            "log10_a": np.log10(float(best_row["scale"])),
        }
        log10_a_best = best["log10_a"]
        log10_a_bounds = (
            (log10_a_best - log10_a_range, log10_a_best + log10_a_range)
            if sample_log10a else
            (-20.0, 0.0)
        )
        prior_config = dict(
            tdust_bounds=(50., 1500.),
            log10_tau_bounds=(-4., 2.),
            log10_a_bounds=log10_a_bounds,
            best=best,
            prior_mode=posterior_mode,
            mix_weight=mix_weight,
            tdust_sigma_frac=tdust_sigma_frac,
            log10_tau_sigma=log10_tau_sigma,
            log10_a_sigma=log10_a_sigma,
            sample_log10a=sample_log10a,
        )
    else:
        ndim = 4 if sample_log10a else 3
        best = {
            "tstar": float(best_row["tstar"]),
            "tdust": float(best_row["tdust"]),
            "log10_tau": np.log10(float(best_row["tau"])),
            "log10_a": np.log10(float(best_row["scale"])),
        }
        log10_a_best = best["log10_a"]
        log10_a_bounds = (
            (log10_a_best - log10_a_range, log10_a_best + log10_a_range)
            if sample_log10a else (-20., 0.)
        )
        prior_config = dict(
            tstar_bounds=(2000., 12000.),
            tdust_bounds=(100., 1500.),
            log10_tau_bounds=(-4., 2.),
            log10_a_bounds=log10_a_bounds,
            best=best,
            prior_mode=posterior_mode,
            mix_weight=mix_weight,
            tstar_sigma_frac=0.3,
            tdust_sigma_frac=tdust_sigma_frac,
            log10_tau_sigma=log10_tau_sigma,
            log10_a_sigma=log10_a_sigma,
            sample_log10a=sample_log10a,
        )


    # # initial positions from best
    # ndim = 4 # tstar, tdust, log10_tau, log10_a
    # p0 = np.zeros((nwalkers, ndim))
    # p0[:, 0] = best["tstar"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    # p0[:, 1] = best["tdust"] * (1.0 + 0.1 * np.random.randn(nwalkers))
    # p0[:, 2] = best["log10_tau"] + 0.2 * np.random.randn(nwalkers)
    # p0[:, 3] = best["log10_a"] + 0.2 * np.random.randn(nwalkers)

    p0 = _initialize_walkers(
        nwalkers=nwalkers,
        best=best,
        prior_config=prior_config,
        ndim=ndim,
        init_mode=init_mode,
        init_scales=init_scales,
        random_seed=random_seed
    )

    if burn_in >= nsteps:
        raise ValueError("burn_in must be less than nsteps.")
    
    ctx = _pick_mp_context(prefer=mp_prefer)
    ncores = min(int(n_cores), nwalkers)

    sampler_kwargs = dict(moves=emcee.moves.StretchMove(a=2.0))

    def _run_sampler(sampler):
        print(f"Running MCMC: ndim={ndim}, nwalkers={nwalkers}, "
          f"nsteps={nsteps}...", flush=True)
        if progress_every is not None:
            state = p0
            for i, state in enumerate(
                    sampler.sample(state, iterations=nsteps), start=1):
                if (i == 1) or (i % progress_every == 0) or (i == nsteps):
                    print(f"[emcee] step {i}/{nsteps}", flush=True)
        else:
            sampler.run_mcmc(p0, nsteps, progress=True)

    if evaluator_mode == "emulator":
        # no pool
        print(f"[pool] emulator mode — running single-threaded (no pool)", flush=True)
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability,
            args=(sed, dusty_runner, prior_config, y_mode, use_weights,
                template, template_tag, tstar_dummy, shell_thickness, sample_log10a, err_floor_frac, equal_weights),
            **sampler_kwargs
        )
        _run_sampler(sampler)
    else:
        print(f"[pool] dusty mode — using {ncores} cores", flush=True)
        with ctx.Pool(processes=ncores) as pool:
            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, log_probability, 
                args=(sed, dusty_runner, prior_config, y_mode, use_weights,
                      template, template_tag, tstar_dummy, shell_thickness, sample_log10a, err_floor_frac, equal_weights),
                pool=pool,
                **sampler_kwargs,
            )
            _run_sampler(sampler)

    # discard burn-in and flatten
    samples = sampler.get_chain(discard=burn_in, flat=True)
    log_prob_samples = sampler.get_log_prob(discard=burn_in, flat=True)

    # for diagnostics
    chain_full = sampler.get_chain() 
    logp_full  = sampler.get_log_prob() 
    af = sampler.acceptance_fraction

    try:
        tau_int = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau_int = None


    # unpack
    if sample_log10a:
        if ndim == 3: # template + sample_log10a
            tdust_samples = samples[:, 0]
            log10_tau_samples = samples[:, 1]
            log10_a_samples = samples[:, 2]
            tstar_samples = np.full_like(tdust_samples, float(tstar_dummy))
        else: # blackbody + sample_log10a
            tstar_samples = samples[:, 0]
            tdust_samples = samples[:, 1]
            log10_tau_samples = samples[:, 2]
            log10_a_samples = samples[:, 3]
    else:
        print("Deriving log10_a analytically for each posterior sample...", flush=True)
        if ndim == 2: # template + analytic a
            tdust_samples = samples[:, 0]
            log10_tau_samples = samples[:, 1]
            tstar_samples = np.full_like(tdust_samples, float(tstar_dummy))
        else: # blackbody + analytic a
            tstar_samples = samples[:, 0]
            tdust_samples = samples[:, 1]
            log10_tau_samples = samples[:, 2]

        phase = sed.get("phase_days", sed.get("phase", None))
        log10_a_samples = np.full(len(samples), np.nan)

        for i, (tdust_i, log10_tau_i) in enumerate(zip(tdust_samples, log10_tau_samples)):
            if i % 500 == 0 or i == len(samples) - 1:
                print(f"  Processing sample {i+1}/{len(samples)}...", flush=True)
            tau_i = 10.0 ** log10_tau_i
            tstar_i = float(tstar_samples[i])
            lam_um, lamFlam, r1 = dusty_runner.evaluate_model(
                tstar=tstar_i,
                tdust=tdust_i,
                tau=tau_i,
                shell_thickness=shell_thickness,
                template=template,
                phase_days=float(phase) if phase is not None else None,
                template_tag=template_tag,
            )
            if lam_um is not None:
                a_i, _ = _ls_scale_and_chi2(
                    lam_um, lamFlam, sed, a=None, y_mode=y_mode, use_weights=use_weights, err_floor_frac=err_floor_frac,
                    equal_weights=equal_weights
                )
                if np.isfinite(a_i) and (a_i > 0):
                    log10_a_samples[i] = np.log10(a_i)
        print(f"  Derived log10_a for {np.sum(np.isfinite(log10_a_samples))}"
                f"/{len(log10_a_samples)} samples.", flush=True)

    tau_samples = 10.0 ** log10_tau_samples
    a_samples   = 10.0 ** np.where(np.isfinite(log10_a_samples),
                                    log10_a_samples, np.nan)

    results = dict(
        samples=samples,
        log_prob=log_prob_samples,
        tstar=tstar_samples,
        tdust=tdust_samples,
        tau=tau_samples,
        a=a_samples,
        grid_best=best,
        prior_config=prior_config,
        sampler=sampler,
        chain=chain_full,
        log_prob_chain=logp_full,
        acceptance_fraction=af,
        autocorr_time=tau_int,
        template_mode=template_mode,
        template_tag=template_tag,
        log10_a_samples=log10_a_samples,
        analytic_log10a=not sample_log10a,
        err_floor_frac=err_floor_frac,
        equal_weights=equal_weights,
    )

    return results