# python -m lltypeiip.emulator.build_template_emulator \
#    --cache-dir dusty_runs/dusty_npz_cache_template \
#    --out-path  dusty_runs/dusty_tmpl_emulator_thick2_0.npz \
#    --thick-tag 2_0 \
#    --template-tag nugent_iip \
#    --n-workers 8

import re
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

def _tau_str_to_float(s):
    idx = s.index("_")
    return float(f"{s[:idx]}.{s[idx+1:]}")

def _parse_filename(fpath, thick_tag="2_0", template_tag="nugent_iip"):
    """
    Tstar_6000_Tdust_1000_tau_0_001_thick_2_0_nugent_iip_phase_100.npz
    Returns (tdust, tau, phase) — tstar is fixed (dummy) in template mode
    """
    s = fpath.stem
    m = re.match(
        r'Tstar_\d+_Tdust_(\d+)_tau_([0-9_]+)_thick_([0-9_]+)_'
        r'(\w+)_phase_(\d+)$',
        s
    )
    if not m:
        return None
    if m.group(3) != thick_tag:
        return None
    if m.group(4) != template_tag:
        return None
    tdust = float(m.group(1))
    tau   = _tau_str_to_float(m.group(2))
    phase = int(m.group(5))
    return tdust, tau, phase

def _load_batch(args):
    batch, lam_ref, n_lam = args
    out = []
    for tdust, tau, phase, fpath in batch:
        try:
            d       = np.load(fpath)
            lam     = d["lam_um"].astype(np.float32)
            flamlam = d["lamFlam"].astype(np.float32)
            if flamlam.max() <= 0:
                continue
            if len(lam) != n_lam:
                flamlam = np.interp(lam_ref, lam, flamlam).astype(np.float32)
            out.append((tdust, tau, phase, flamlam))
        except Exception:
            continue
    return out

def main():

    p = argparse.ArgumentParser(
        description="Build template RegularGrid emulator from cached DUSTY npz files."
    )
    p.add_argument("--cache-dir",    default="dusty_runs/dusty_npz_cache_template")
    p.add_argument("--output-path",     default="dusty_runs/dusty_tmpl_emulator_thick2_0.npz")
    p.add_argument("--thick-tag",    default="2_0")
    p.add_argument("--template-tag", default="nugent_iip")
    p.add_argument("--n-workers",    type=int, default=8)
    p.add_argument("--io-batch",     type=int, default=500)
    args = p.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("TEMPLATE EMULATOR — BUILD (RegularGridInterpolator)")
    print("=" * 60)
    print(f"  cache_dir    : {cache_dir}")
    print(f"  out_path     : {output_path}")
    print(f"  thick_tag    : {args.thick_tag}")
    print(f"  template_tag : {args.template_tag}")
    print()

    # first scan files
    print("Scanning files...")
    t0      = time.time()
    records = []
    for f in sorted(cache_dir.glob(
            f"*thick_{args.thick_tag}*{args.template_tag}*.npz")):
        parsed = _parse_filename(f, args.thick_tag, args.template_tag)
        if parsed is not None:
            records.append((*parsed, f))
    print(f"  {len(records):,} files in {time.time()-t0:.1f}s")

    # grid axes
    tdust_u = np.array(sorted({r[0] for r in records}))
    tau_u   = np.array(sorted({r[1] for r in records}))
    phase_u = np.array(sorted({r[2] for r in records}))
    log10tau_u = np.log10(tau_u)

    print(f"\n  T_dust : {len(tdust_u)} values  {tdust_u}")
    print(f"  tau    : {len(tau_u)} values  [{tau_u[0]:.4f}, {tau_u[-1]:.2f}]")
    print(f"  phase  : {len(phase_u)} values  [{phase_u[0]}, {phase_u[-1]}]")
    n_exp = len(tdust_u) * len(tau_u) * len(phase_u)
    print(f"  Expected: {n_exp:,}  Actual: {len(records):,}  "
          f"({'✓ complete' if n_exp == len(records) else '✗ incomplete'})")
    
    # ref wavelength grid
    probe   = np.load(records[0][3])
    lam_ref = probe["lam_um"].astype(np.float32)
    n_lam   = len(lam_ref)
    print(f"\n  λ: {n_lam} pts  [{lam_ref[0]:.4f}, {lam_ref[-1]:.2f}] µm")

    mem_gb = len(tdust_u)*len(tau_u)*len(phase_u)*n_lam*4/1e9
    print(f"  SED cube: {mem_gb*1000:.1f} MB (float32)")

    # load data in batches
    tdust_idx = {v: i for i, v in enumerate(tdust_u)}
    tau_idx   = {v: i for i, v in enumerate(tau_u)}
    phase_idx = {v: i for i, v in enumerate(phase_u)}

    batches = [
        (records[i:i+args.io_batch], lam_ref, n_lam)
        for i in range(0, len(records), args.io_batch)
    ]

    sed_cube = np.full(
        (len(tdust_u), len(tau_u), len(phase_u), n_lam),
        np.nan, dtype=np.float32
    )

    print(f"\nLoading {len(records):,} files "
          f"({len(batches)} batches, {args.n_workers} workers)...")
    t0 = time.time()
    n_loaded = n_errors = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futs = [exe.submit(_load_batch, b) for b in batches]
        for i, fut in enumerate(as_completed(futs)):
            for tdust, tau, phase, flamlam in fut.result():
                ii = tdust_idx.get(tdust)
                jj = tau_idx.get(tau)
                kk = phase_idx.get(phase)
                if None in (ii, jj, kk):
                    n_errors += 1
                    continue
                sed_cube[ii, jj, kk, :] = flamlam
                n_loaded += 1
            if (i+1) % 5 == 0 or (i+1) == len(batches):
                print(f"  [{i+1:3d}/{len(batches)}]  "
                      f"loaded={n_loaded:,}  err={n_errors}",
                      flush=True)
                
    nan_frac = np.isnan(sed_cube).mean()
    print(f"\nLoaded {n_loaded:,}  NaN={nan_frac:.2%}  "
          f"time={time.time()-t0:.1f}s")

    if nan_frac > 0.01:
        print("WARNING: >1% NaN — grid may be incomplete")
    
    # save the result
    print(f"\nSaving to {output_path}...")
    np.savez_compressed(
        output_path,
        lam_ref    = lam_ref,
        sed_cube   = sed_cube,        # (n_tdust, n_tau, n_phase, n_lam)
        tdust_u    = tdust_u,
        tau_u      = tau_u,
        log10tau_u = log10tau_u,
        phase_u    = phase_u,
        thick_tag  = np.array([args.thick_tag]),
        template_tag = np.array([args.template_tag]),
    )
    print(f"Saved {output_path.stat().st_size/1e6:.1f} MB")
    print(f"\nDone. Load with:")
    print(f"  from lltypeiip.emulator import DustyTemplateEmulator")
    print(f"  emu = DustyTemplateEmulator('{output_path}')")

if __name__ == "__main__":
    main()