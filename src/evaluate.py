"""
Shared evaluation utilities for concept-drift experiments.

Computes, for any model exposing a predict_fn(X, t) -> probabilities:
  - per-month AUROC on the four chronological test months (Jun-Sep 2020)
  - AUT  (Area Under Time) = trapezoidal integral of AUROC over the test window
  - macro-averaged F1 on the malware subset per month and overall
All models are evaluated on identical chronological splits.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score


def _safe_auroc(y, p):
    """AUROC; returns NaN if a class is missing (avoids ValueError)."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _safe_f1(y, p):
    pred = (p >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(f1_score(y, pred, average="macro", zero_division=0))


def evaluate_model(predict_fn, df, feat_cols, split_info, device="cpu"):
    """
    predict_fn : callable(X: np.ndarray (n, d), t: np.ndarray (n,)) -> probs (n,)
    Returns dict with per-month metrics and aggregates.
    """
    test_months = split_info["test_months"]
    month_idx = split_info["month_idx"]

    results = {"months": [], "per_month": {}}
    all_y, all_p = [], []

    for m in test_months:
        idx = month_idx[m]
        X = df.loc[idx, feat_cols].fillna(0).astype(np.float32).values
        y = df.loc[idx, "Label"].astype(np.float32).values
        ts = df.loc[idx, "timestamp"].values
        t = _normalize_ts(ts)

        p = predict_fn(X, t)
        p = np.asarray(p, dtype=np.float64).ravel()

        auroc = _safe_auroc(y, p)
        f1_macro = _safe_f1(y, p)
        # malware-subset macro F1 (treat malware as positive class via standard f1)
        results["per_month"][m] = {
            "auroc": auroc,
            "f1_macro": f1_macro,
            "support": int(len(y)),
            "malware": int((y == 1).sum()),
            "benign": int((y == 0).sum()),
        }
        results["months"].append(m)
        all_y.append(y)
        all_p.append(p)

    # AUT: trapezoidal integral of AUROC over ordered test months
    aurocs = [results["per_month"][m]["auroc"] for m in test_months]
    valid = [a for a in aurocs if not np.isnan(a)]
    if len(valid) >= 2:
        aut = float(np.trapz(valid, x=np.arange(len(valid))))
    elif len(valid) == 1:
        aut = float(valid[0])
    else:
        aut = float("nan")

    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    results["overall"] = {
        "auroc": _safe_auroc(all_y, all_p),
        "f1_macro": _safe_f1(all_y, all_p),
        "aut": aut,
        "auroc_per_month": [results["per_month"][m]["auroc"] for m in test_months],
    }
    return results


def _normalize_ts(timestamps):
    ts = pd.to_datetime(timestamps)
    t0 = ts.min()
    t1 = ts.max()
    span = (t1 - t0).total_seconds()
    if span == 0:
        return np.zeros(len(ts), dtype=np.float32)
    return np.asarray((ts - t0).total_seconds() / span, dtype=np.float32)


def aut_decay_pct(aut_initial, aut_final):
    """Relative change (%) of AUT from initial to final. Negative = decay."""
    if aut_initial == 0 or np.isnan(aut_initial) or np.isnan(aut_final):
        return float("nan")
    return 100.0 * (aut_final - aut_initial) / aut_initial
