"""
Latent Neural ODE training + month-by-month evaluation.

Trains the continuous-time model on the chronological train split and
evaluates AUROC / AUT month-by-month on Jun-Sep 2020. Uses the adjoint
sensitivity method (odeint_adjoint) for constant-memory backpropagation
and Focal Loss for class-imbalance robustness.

Solver: dopri5 (adaptive, default) or rk4 (fixed-step) via --solver.
Focal Loss: on by default (--loss bce to switch).
--decoupled disables the ODE solver (decoupled encoder/decoder ablation).

Outputs:
  - results/ode_metrics_<config>.csv
  - results/ode_model_<config>.pt
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from chronology import load_dataframe, split_chronological, get_arrays, normalize_time
from evaluate import evaluate_model
from ode_model import LatentNeuralODE, FocalLoss

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cpu"
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def subsample(X, y, t, max_samples, random_state=42):
    n = X.shape[0]
    if n <= max_samples:
        return X, y, t
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X[idx], y[idx], t[idx]


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n_batches = 0.0, 0
    for Xb, yb, tb in loader:
        Xb, yb, tb = Xb.to(device), yb.to(device), tb.to(device)
        optimizer.zero_grad()
        logits, _, _ = model(Xb, tb)
        loss = criterion(logits, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_loss(model, Xv, yv, tv, criterion, device, batch_size=1024):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, Xv.shape[0], batch_size):
        Xb = torch.from_numpy(Xv[i:i + batch_size]).to(device)
        yb = torch.from_numpy(yv[i:i + batch_size]).to(device)
        tb = torch.from_numpy(tv[i:i + batch_size]).to(device)
        logits, _, _ = model(Xb, tb)
        total += float(criterion(logits, yb))
        n += 1
    return total / max(n, 1)


def make_ode_predict_fn(model, device):
    def predict_fn(X, t):
        model.eval()
        with torch.no_grad():
            Xb = torch.from_numpy(X.astype(np.float32)).to(device)
            tb = torch.from_numpy(t.astype(np.float32)).to(device)
            logits, _, _ = model(Xb, tb)
            prob = torch.sigmoid(logits).cpu().numpy()
        return prob
    return predict_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--solver", type=str, default="dopri5", choices=["dopri5", "rk4"])
    ap.add_argument("--loss", type=str, default="focal", choices=["focal", "bce"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-samples", type=int, default=15000)
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--decoupled", action="store_true",
                    help="disable the ODE solver (decoupled encoder/decoder ablation)")
    args = ap.parse_args()

    df, feat_cols = load_dataframe(max_features=600)
    split_info = split_chronological(df)

    Xtr, ytr, ttr = get_arrays(df, feat_cols, split_info["train"])
    Xtr, ytr, ttr = subsample(Xtr, ytr, ttr, args.max_train_samples)
    ttr = normalize_time(ttr)

    Xv, yv, tv = get_arrays(df, feat_cols, split_info["val"])
    tv = normalize_time(tv)

    input_dim = Xtr.shape[1]
    model = LatentNeuralODE(
        input_dim=input_dim, latent_dim=args.latent_dim, hidden=args.hidden,
        solver=args.solver, use_ode=not args.decoupled, atol=args.atol, rtol=args.rtol,
    ).to(DEVICE)

    criterion = FocalLoss() if args.loss == "focal" else torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = TensorDataset(
        torch.from_numpy(Xtr), torch.from_numpy(ytr), torch.from_numpy(ttr),
    )
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model_label = ("Decoupled encoder/decoder (no ODE)" if args.decoupled
                   else f"Neural ODE ({args.solver}, {args.loss})")
    config_tag = ("decoupled" if args.decoupled else f"{args.solver}_{args.loss}")
    metrics_path = os.path.join(RESULTS_DIR, f"ode_metrics_{config_tag}.csv")
    ckpt_path = os.path.join(RESULTS_DIR, f"ode_model_{config_tag}.pt")
    print(f"[INFO] Training {model_label} (latent={args.latent_dim}, hidden={args.hidden}) "
          f"on {Xtr.shape[0]} samples, {input_dim} features, device={DEVICE}")
    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, loader, optimizer, criterion, DEVICE)
        vl_loss = val_loss(model, Xv, yv, tv, criterion, DEVICE)
        print(f"     epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={vl_loss:.4f}")
        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), ckpt_path)

    # Month-by-month evaluation on the test window
    predict_fn = make_ode_predict_fn(model, DEVICE)
    res = evaluate_model(predict_fn, df, feat_cols, split_info, device=DEVICE)
    ov = res["overall"]
    print(f"\n[INFO] {model_label} Overall AUROC: {ov['auroc']:.4f}  Macro F1: {ov['f1_macro']:.4f}  AUT: {ov['aut']:.4f}")
    for m in res["months"]:
        pm = res["per_month"][m]
        print(f"     {m}: AUROC={pm['auroc']:.4f} F1={pm['f1_macro']:.4f} "
              f"(n={pm['support']}, malware={pm['malware']})")

    # Save metrics CSV
    rows = []
    for m in res["months"]:
        pm = res["per_month"][m]
        rows.append({
            "model": model_label, "month": m,
            "auroc": pm["auroc"], "f1_macro": pm["f1_macro"],
            "support": pm["support"], "malware": pm["malware"], "benign": pm["benign"],
        })
    rows.append({
        "model": model_label, "month": "OVERALL",
        "auroc": ov["auroc"], "f1_macro": ov["f1_macro"],
        "support": rows[0]["support"] + rows[0]["benign"] if rows else 0,
        "malware": sum(r["malware"] for r in rows), "benign": sum(r["benign"] for r in rows),
        "aut": ov["aut"],
    })
    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"[INFO] {model_label} metrics written to {metrics_path}")
    print(f"[INFO] Model checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    main()
