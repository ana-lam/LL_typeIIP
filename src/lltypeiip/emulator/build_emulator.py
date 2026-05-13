# python -m lltypeiip.inference.build_emulator --cache-dir dusty_runs/dusty_npz_cache_blackbody \
# --output-path dusty_runs/dusty_nn_emulator_bb_thick2_0.npz --max-models 300000 --epochs 300 \
# --hidden 512 --depth 5 --n-workers 16

import re
import time
import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .emulator import _build_mlp


# Filename parsing from cache

def _tau_str_to_float(s):
    """'0_0001' → 0.0001,  '1_0' → 1.0,  '10_0' → 10.0"""
    idx = s.index("_")
    return float(f"{s[:idx]}.{s[idx+1:]}")

def _parse_filename(fpath, thick_tag="2_0"):
    m = re.match(
        r'Tstar_(\d+)_Tdust_(\d+)_tau_([0-9_]+)_thick_([0-9_]+)$',
        fpath.stem
    )
    if not m or m.group(4) != thick_tag:
        return None
    return (
        float(m.group(1)),
        float(m.group(2)),
        _tau_str_to_float(m.group(3)),
    )

# Batch loading
def _load_batch(args):
    """Load a batch of (tstar, tdust, tau, fpath) records."""
    batch, lam_ref, n_lam = args
    out = []

    for tstar, tdust, tau, fpath in batch:
        try:
            d = np.load(fpath)
            lam = d["lam_um"].astype(np.float32)
            flamlam = d["lamFlam"].astype(np.float32)

            if flamlam.max() <= 0:
                continue

            if len(lam) != n_lam or not np.allclose(lam, lam_ref, rtol=1e-3):
                flamlam = np.interp(lam_ref, lam, flamlam).astype(np.float32)
            
            out.append((tstar, tdust, tau, flamlam))
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return out

def main():
    p = argparse.ArgumentParser(
        description="Build DustyNNEmulator from cached DUSTY npz files."
    )

    p.add_argument("--cache-dir", default = "dusty_runs/dusty_npz_cache_blackbody")
    p.add_argument("--output-path", default = "dusty_runs/dusty_nn_emulator_bb_thick2_0.npz")
    p.add_argument("--thick-tag", default="2_0", help="The shell thickness tag to filter the cached files.")
    p.add_argument("--max-models", type=int, default=300_000, 
        help="Maximum number of models to include in the emulator.")
    
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of the dataset to use for validation.")
    
    p.add_argument("--n-workers", type=int, default=4,
                   help="Number of worker processes to use for data loading.")
    p.add_argument("--io-batch",    type=int, default=2000,
                   help="Files per I/O batch")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    
    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 65)
    print("DUSTY NN EMULATOR — BUILD")
    print("=" * 65)
    print(f"  cache_dir  : {cache_dir}")
    print(f"  out_path   : {output_path}")
    print(f"  thick_tag  : {args.thick_tag}")
    print(f"  max_models : {args.max_models}")
    print(f"  hidden/depth: {args.hidden} / {args.depth}")
    print(f"  epochs     : {args.epochs}")
    print(f"  device     : {device}")
    print()

    # Step 1: scan filenames in cache
    print("Scanning cache directory...")

    t0 = time.time()
    records = []

    for f in cache_dir.glob(f"*thick_{args.thick_tag}*.npz"):
        parsed = _parse_filename(f, thick_tag=args.thick_tag)
        if parsed is not None:
            records.append((*parsed, f))

    print(f"  Found {len(records)} records in {time.time() - t0:.2f} seconds.")

    if len(records) == 0:
        raise RuntimeError("No valid records found in the cache directory.")
    
    # Step 2: subsample the records
    if args.max_models > 0 and len(records) > args.max_models:
        idx = rng.choice(len(records), size=args.max_models, replace=False)
        records = [records[i] for i in sorted(idx)]
        print(f"  Subsampled to {len(records)} records.")

    # Step 3: get reference wavelength grid
    probe = np.load(records[0][3])
    lam_ref = probe["lam_um"].astype(np.float32)
    n_lam = len(lam_ref)
    print(f"  Reference wavelength grid: {n_lam} points.")

    # Step 4: load the data in batches
    batches = [
        (records[i:i+args.io_batch], lam_ref, n_lam)
        for i in range(0, len(records), args.io_batch)
    ]

    print(f"\nLoading {len(records):,} SEDs "
          f"({len(batches)} batches, {args.n_workers} workers)...")
    
    X_list = [] # (tstar, tdust, log10tau)
    Y_list = [] # log(flamlam)

    t0 = time.time()
    n_loaded = n_skip = 0

    with ProcessPoolExecutor(max_workers=args.n_workers) as exe:
        futs = [exe.submit(_load_batch, batch) for batch in batches]
        for i, fut in enumerate(as_completed(futs)):
            for tstar, tdust, tau, flamlam in fut.result():
                log10_tau = np.log10(max(tau, 1e-10))
                X_list.append([tstar, tdust, log10_tau])
                # Log-transform SED — values are ~O(1) so log is well-behaved
                Y_list.append(np.log(np.clip(flamlam, 1e-10, None)))
                n_loaded += 1
        
        n_done = i + 1
        if n_done % 20 == 0 or n_done == len(batches):
            rate = n_loaded / max(time.time()-t0, 1e-6)
            eta = (len(records) - n_loaded) / max(rate, 1)
            print(f"  [{n_done:4d}/{len(batches)}]  "
                      f"loaded={n_loaded:,}  "
                      f"skip={n_skip}  "
                      f"{rate:,.0f}/s  "
                      f"ETA={eta/60:.1f}min",
                      flush=True)
            
    print(f"\nLoaded {n_loaded:,} models in {(time.time()-t0)/60:.1f}min")

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    print(f"X: {X.shape}  Y: {Y.shape}")
    print(f"X ranges:  "
          f"tstar=[{X[:,0].min():.0f},{X[:,0].max():.0f}]  "
          f"tdust=[{X[:,1].min():.0f},{X[:,1].max():.0f}]  "
          f"log10tau=[{X[:,2].min():.2f},{X[:,2].max():.2f}]")
    
    # Step 5: normalize
    X_mean = X.mean(axis=0).astype(np.float64)
    X_std  = X.std(axis=0).astype(np.float64)
    Y_mean = Y.mean(axis=0).astype(np.float64)
    Y_std  = Y.std(axis=0).astype(np.float64)
    Y_std  = np.where(Y_std > 0, Y_std, 1.0)

    X_norm = ((X - X_mean) / X_std).astype(np.float32)
    Y_norm = ((Y - Y_mean) / Y_std).astype(np.float32)

    # Step 6: train/val split
    n_val   = max(1, int(len(X_norm) * args.val_frac))
    val_idx = rng.choice(len(X_norm), n_val, replace=False)
    tr_mask = np.ones(len(X_norm), dtype=bool)
    tr_mask[val_idx] = False

    Xtr = torch.tensor(X_norm[tr_mask])
    Ytr = torch.tensor(Y_norm[tr_mask])
    Xva = torch.tensor(X_norm[~tr_mask])
    Yva = torch.tensor(Y_norm[~tr_mask])
    print(f"\nTrain: {len(Xtr):,}   Val: {len(Xva):,}")

    tr_loader = DataLoader(
        TensorDataset(Xtr, Ytr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Step 7: train
    print(f"\nTraining on {device}  "
          f"(hidden={args.hidden}, depth={args.depth}, "
          f"epochs={args.epochs}, lr={args.lr})...")

    model = _build_mlp(n_in=3, n_out=n_lam,
                       hidden=args.hidden, depth=args.depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs
    )

    Xva_d = Xva.to(device)
    Yva_d = Yva.to(device)

    best_val = np.inf
    best_state = None
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_losses = []
        for Xb, Yb in tr_loader:
            Xb, Yb = Xb.to(device), Yb.to(device)
            loss = ((model(Xb) - Yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        sched.step()

        # validation
        model.eval()
        with torch.no_grad():
            val_loss = ((model(Xva_d) - Yva_d) ** 2).mean().item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"  Epoch {epoch:4d}/{args.epochs}  "
                  f"train={np.mean(tr_losses):.5f}  "
                  f"val={val_loss:.5f}  "
                  f"best_val={best_val:.5f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}  "
                  f"elapsed={time.time()-t0:.0f}s")

    print(f"\nBest val loss: {best_val:.5f}")
    model.load_state_dict(best_state)

    # Step 8: final validation
    model.eval()
    with torch.no_grad():
        Y_pred_norm = model(Xva.to(device)).cpu().numpy()

    Y_pred_log = Y_pred_norm * Y_std + Y_mean
    Y_true_log = Y_norm[~tr_mask] * Y_std + Y_mean

    # Back to linear flux
    Y_pred_flux = np.exp(Y_pred_log)
    Y_true_flux = np.exp(Y_true_log)

    rel_err = np.abs(Y_pred_flux - Y_true_flux) / (np.abs(Y_true_flux) + 1e-10)
    med_rel = np.median(rel_err)
    p90_rel = np.percentile(rel_err, 90)
    p99_rel = np.percentile(rel_err, 99)

    print(f"\nValidation flux errors (relative, {len(Xva):,} models):")
    print(f"  Median : {med_rel:.4f}  ({med_rel*100:.2f}%)")
    print(f"  90th % : {p90_rel:.4f}  ({p90_rel*100:.2f}%)")
    print(f"  99th % : {p99_rel:.4f}  ({p99_rel*100:.2f}%)")

    if med_rel < 0.01:
        print("  ✓ Excellent  (median < 1%)")
    elif med_rel < 0.05:
        print("  ~ Acceptable (median < 5%) — consider more epochs or larger network")
    else:
        print("  ✗ Poor       (median > 5%) — increase --epochs or --hidden")

    # Step 9: save the model

    print(f"\nSaving to {output_path}...")
    save_dict = dict(
        lam_ref = lam_ref,
        X_mean  = X_mean,
        X_std   = X_std,
        Y_mean  = Y_mean,
        Y_std   = Y_std,
        X_train = X,          # kept for bounds checking in emulator
    )

    # Store weights with "w__" prefix, dots → double underscores
    for k, v in best_state.items():
        save_dict[f"w__{k.replace('.', '__')}"] = v.numpy()

    np.savez_compressed(output_path, **save_dict)
    sz = output_path.stat().st_size / 1e6
    print(f"Saved {sz:.1f} MB")
    print(f"\nDone. Load with:")
    print(f"  from lltypeiip.inference.emulator import DustyNNEmulator")
    print(f"  emu = DustyNNEmulator('{output_path}')")

if __name__ == "__main__":
    main()