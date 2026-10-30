"""
Generate the seven capstone deliverable figures from real data.

Outputs (one sub-folder per figure under graphs/):
  01_temporal_split/       temporal split + sample distribution
  02_feature_drift/        top drifting PE features
  03_novel_vs_persistent/  novel vs persistent malware families
  04_roc_curves/           full ROC + low-FPR zoom
  05_f1_recall_decay/      monthly macro-F1 / recall decay
  06_confusion_matrices/   confusion matrix heatmaps
  07_system_architecture/  Fig.1 architecture + Fig.2 workflow

Predictions are produced from the checkpoints in notebooks/results/ (the same
models that generated notebooks/results/*_metrics.csv), using the exact feature
order each model was trained with:
  - static baselines : top-600 drift features, file order (stored in joblib)
  - Neural ODE / ablation : top-600 drift features, drift-ranking order
"""

import os
import re
import sys
import glob

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, recall_score
from sklearn.metrics import precision_score, confusion_matrix

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import torch
from ode_model import LatentNeuralODE

PARQUET = os.path.join(ROOT, "dataset", "bodmas_clean_features.parquet")
DRIFT_CSV = os.path.join(ROOT, "dataset", "feature_drift_ranking.csv")
RESULTS = os.path.join(ROOT, "notebooks", "results")
BASELINE_DIR = os.path.join(RESULTS, "baseline_models")
NOVEL_TXT = os.path.join(ROOT, "drive-download-20261026T224800Z-1-001",
                         "novel_family_analysis.txt")
GRAPH_DIR = os.path.join(ROOT, "graphs")
DERIVED_DIR = os.path.join(GRAPH_DIR, "_derived")
PRED_CACHE = os.path.join(DERIVED_DIR, "test_predictions.npz")

os.makedirs(GRAPH_DIR, exist_ok=True)
os.makedirs(DERIVED_DIR, exist_ok=True)

TRAIN_MONTHS = [f"{y}-{m:02d}" for y, m in
                [(2019, m) for m in range(8, 13)] + [(2020, m) for m in range(1, 4)]]
VAL_MONTHS = ["2020-04", "2020-05"]
TEST_MONTHS = ["2020-06", "2020-07", "2020-08", "2020-09"]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": ":",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {
    "XGBoost (static)": "#d62728",
    "LightGBM (static)": "#ff7f0e",
    "MLP (static)": "#9467bd",
    "Neural ODE (dopri5)": "#2ca02c",
    "Neural ODE (rk4)": "#1f77b4",
    "Decoupled (no ODE)": "#7f7f7f",
}
STYLE = {
    "XGBoost (static)": dict(ls="--", lw=2.0),
    "LightGBM (static)": dict(ls="--", lw=2.0),
    "MLP (static)": dict(ls="--", lw=2.0),
    "Neural ODE (dopri5)": dict(ls="-", lw=2.2),
    "Neural ODE (rk4)": dict(ls="-", lw=2.2),
    "Decoupled (no ODE)": dict(ls=":", lw=2.0),
}
DISPLAY_ORDER = ["XGBoost (static)", "LightGBM (static)", "MLP (static)",
                 "Neural ODE (rk4)", "Neural ODE (dopri5)"]


def _out(sub):
    d = os.path.join(GRAPH_DIR, sub)
    os.makedirs(d, exist_ok=True)
    return d


def _savefig(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[INFO] saved", os.path.relpath(path, ROOT))


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_data():
    xgb = joblib.load(os.path.join(BASELINE_DIR, "xgboost_(static).joblib"))
    baseline_feats = list(xgb["feat_cols"])

    rank = pd.read_csv(DRIFT_CSV)
    drift_all = rank.iloc[:, 0].astype(str).tolist()
    import pyarrow.parquet as pq
    available = set(pq.ParquetFile(PARQUET).schema_arrow.names)
    drift_feats = [c for c in drift_all if c in available][:600]

    cols = ["sha256", "Label", "family", "timestamp"] + baseline_feats
    df = pd.read_parquet(PARQUET, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M").astype(str)
    return df, baseline_feats, drift_feats


def split_of(month):
    if month in TRAIN_MONTHS:
        return "Train"
    if month in VAL_MONTHS:
        return "Val"
    if month in TEST_MONTHS:
        return "Test"
    return "Other"


# ----------------------------------------------------------------------------
# 1. Temporal split + sample distribution
# ----------------------------------------------------------------------------

def fig_temporal_split(df):
    out = _out("01_temporal_split")
    counts = (df.groupby(["month", "Label"]).size().unstack(fill_value=0)
              .reindex(columns=[0, 1], fill_value=0))
    counts.columns = ["Benign", "Malware"]
    counts = counts.sort_index()
    counts["Total"] = counts.sum(axis=1)
    counts["Split"] = [split_of(m) for m in counts.index]
    counts.index.name = "month"
    counts.to_csv(os.path.join(out, "temporal_split_counts.csv"))

    months = list(counts.index)
    benign = counts["Benign"].values
    malware = counts["Malware"].values
    splits = counts["Split"].values

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(months))
    b1 = ax.bar(x, benign, color="#4c9be8", label="Benign", width=0.78,
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x, malware, bottom=benign, color="#d9534f", label="Malware",
                width=0.78, edgecolor="white", linewidth=0.5)

    ymax = counts["Total"].max() * 1.28
    ax.set_ylim(0, ymax)

    span_colors = {"Train": "#eaf2fb", "Val": "#fff6e6", "Test": "#eaf7ee"}
    spans = {"Train": [], "Val": [], "Test": []}
    for i, s in enumerate(splits):
        if s in spans:
            spans[s].append(i)
    for s, idxs in spans.items():
        if not idxs:
            continue
        ax.axvspan(min(idxs) - 0.5, max(idxs) + 0.5, color=span_colors[s],
                   zorder=0, alpha=0.85)
        ax.text(np.mean(idxs), ymax * 0.98, s,
                ha="center", va="top", fontsize=13, fontweight="bold",
                color={"Train": "#2b5d8a", "Val": "#a36a00", "Test": "#2e7d4f"}[s])

    for i, (b, m) in enumerate(zip(benign, malware)):
        ax.text(i, b + m + ymax * 0.012,
                f"{int(b + m):,}", ha="center", va="bottom", fontsize=7.5,
                rotation=90, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels([pd.Period(m).strftime("%b\n%Y") for m in months], fontsize=8.5)
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("Calendar month")
    ax.set_title("Temporal dataset split and class distribution (BODMAS, Aug 2019 - Sep 2020)",
                 fontsize=13, pad=12)
    ax.legend(loc="upper left", framealpha=0.95)

    tot_tr = counts[counts["Split"] == "Train"]["Total"].sum()
    tot_va = counts[counts["Split"] == "Val"]["Total"].sum()
    tot_te = counts[counts["Split"] == "Test"]["Total"].sum()
    ax.text(0.995, 0.72,
            f"Train: {tot_tr:,}\nVal:   {tot_va:,}\nTest:  {tot_te:,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc", alpha=0.9))
    _savefig(fig, os.path.join(out, "temporal_split_distribution.png"))


# ----------------------------------------------------------------------------
# 2. Top drifting features
# ----------------------------------------------------------------------------

def fig_top_drift():
    out = _out("02_feature_drift")
    rank = pd.read_csv(DRIFT_CSV)
    rank.columns = ["feature", "drift_magnitude"]
    top = rank.head(20).iloc[::-1].reset_index(drop=True)
    top.to_csv(os.path.join(out, "top20_drifting_features.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(top)))
    bars = ax.barh(top["feature"], top["drift_magnitude"], color=colors,
                   edgecolor="black", linewidth=0.4, height=0.78)
    for b, v in zip(bars, top["drift_magnitude"]):
        ax.text(v + 0.003, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=8)
    ax.set_xlabel("Drift magnitude (Wasserstein / KS ranking score)")
    ax.set_ylabel("PE feature")
    ax.set_title("Top 20 drifting PE features (train vs test window)", fontsize=13)
    ax.set_xlim(0, top["drift_magnitude"].max() * 1.15)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.grid(False, axis="y")
    _savefig(fig, os.path.join(out, "top_drifting_features.png"))


# ----------------------------------------------------------------------------
# 3. Novel vs persistent families
# ----------------------------------------------------------------------------

def _parse_novel_txt(path):
    txt = open(path, "r", encoding="utf-8").read()
    def grab(label, cast=float):
        m = re.search(re.escape(label) + r":\s*([0-9.]+)", txt)
        return cast(m.group(1)) if m else float("nan")
    info = {
        "novel_families": int(grab("Novel Families Count", float)),
        "novel_samples": int(grab("Novel Family Test Samples", float)),
        "persistent_families": int(grab("Persistent Families Count", float)),
        "persistent_samples": int(grab("Persistent Test Samples", float)),
        "auroc_persistent": grab("AUROC (Test_Persistent)"),
        "auroc_novel": grab("AUROC (Test_Novel)"),
        "auroc_gap": grab("AUROC Performance Gap"),
    }
    pat = re.compile(r"^\s*-\s+(\S+)\s*:\s*Samples=(\d+)\s*\|\s*AUROC=([0-9.]+)")
    novel, persistent = [], []
    section = None
    for line in txt.splitlines():
        if "TOP 10 NOVEL" in line:
            section = novel
            continue
        if "TOP 5 PERSISTENT" in line:
            section = persistent
            continue
        m = pat.match(line)
        if m and section is not None:
            section.append({"family": m.group(1), "samples": int(m.group(2)),
                            "auroc": float(m.group(3))})
    info["top_novel"] = novel
    info["top_persistent"] = persistent
    return info


def fig_novel(info):
    out = _out("03_novel_vs_persistent")
    rows = [{"group": "Novel", "families": info["novel_families"],
             "test_malware_samples": info["novel_samples"],
             "auroc": info["auroc_novel"]},
            {"group": "Persistent", "families": info["persistent_families"],
             "test_malware_samples": info["persistent_samples"],
             "auroc": info["auroc_persistent"]}]
    pd.DataFrame(rows).to_csv(os.path.join(out, "novel_vs_persistent_summary.csv"),
                              index=False)
    pd.DataFrame(info["top_novel"]).to_csv(
        os.path.join(out, "top_novel_families.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    c_novel, c_pers = "#d9534f", "#4c9be8"
    groups = ["Novel", "Persistent"]
    gcol = [c_novel, c_pers]

    ax = axes[0, 0]
    vals = [info["novel_families"], info["persistent_families"]]
    bars = ax.bar(groups, vals, color=gcol, width=0.55, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Malware families in test window")
    ax.set_ylabel("Number of families")
    ax.margins(y=0.15)

    ax = axes[0, 1]
    vals = [info["novel_samples"], info["persistent_samples"]]
    bars = ax.bar(groups, vals, color=gcol, width=0.55, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, f"{v:,}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Test malware samples")
    ax.set_ylabel("Samples")
    ax.margins(y=0.15)

    ax = axes[1, 0]
    vals = [info["auroc_novel"], info["auroc_persistent"]]
    bars = ax.bar(groups, vals, color=gcol, width=0.55, edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.00015, f"{v:.4f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(max(0.99, min(vals) - 0.002), 1.001)
    ax.set_ylabel("AUROC")
    ax.set_title(f"Detection on unseen vs known families\n"
                 f"(gap = {info['auroc_gap']:.4f})")

    ax = axes[1, 1]
    top = info["top_novel"][::-1]
    names = [t["family"] for t in top]
    samples = [t["samples"] for t in top]
    bars = ax.barh(names, samples, color=c_novel, alpha=0.85,
                   edgecolor="black", linewidth=0.4)
    for b, t in zip(bars, top):
        ax.text(b.get_width() + max(samples) * 0.02,
                b.get_y() + b.get_height() / 2,
                f"n={t['samples']}, AUC={t['auroc']:.4f}",
                va="center", fontsize=8)
    ax.set_xlabel("Test samples")
    ax.set_title("Top 10 unseen (novel) families by sample count")
    ax.set_xlim(0, max(samples) * 1.55)
    ax.grid(True, axis="x", alpha=0.3, linestyle=":")
    ax.grid(False, axis="y")

    fig.suptitle("Novel vs persistent malware family analysis", fontsize=15, y=1.00)
    plt.tight_layout()
    _savefig(fig, os.path.join(out, "novel_vs_persistent_analysis.png"))


# ----------------------------------------------------------------------------
# Predictions for ROC / F1-recall / confusion
# ----------------------------------------------------------------------------

def _model_specs():
    return [
        dict(key="xgboost", label="XGBoost (static)", kind="baseline",
             file="xgboost_(static).joblib"),
        dict(key="lightgbm", label="LightGBM (static)", kind="baseline",
             file="lightgbm_(static).joblib"),
        dict(key="mlp", label="MLP (static)", kind="baseline",
             file="mlp_(static).joblib"),
        dict(key="ode_rk4", label="Neural ODE (rk4)", kind="ode",
             file="ode_model_rk4_focal.pt", solver="rk4", use_ode=True),
        dict(key="ode_dopri5", label="Neural ODE (dopri5)", kind="ode",
             file="ode_model_dopri5_focal.pt", solver="dopri5", use_ode=True),
    ]


def _predict_baseline(model, scaler, X):
    Xf = scaler.transform(X) if scaler is not None else X
    return model.predict_proba(Xf)[:, 1].astype(np.float64)


def _predict_ode(model, X, t, batch=4096):
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            tb = torch.from_numpy(t[i:i + batch])
            logits, _, _ = model(xb, tb)
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out).astype(np.float64)


def build_predictions(df, baseline_feats, drift_feats):
    specs = _model_specs()
    if os.path.exists(PRED_CACHE):
        d = np.load(PRED_CACHE, allow_pickle=True)
        meta = {k: d[k] for k in d.files if k in ("y", "month", "family", "sha256")}
        cached = {k.replace("p_", ""): d[k] for k in d.files if k.startswith("p_")}
        preds = {s["key"]: cached[s["key"]] for s in specs if s["key"] in cached}
        if len(preds) == len(specs):
            print("[INFO] loaded cached test predictions")
            return preds, meta

    test = df[df["month"].isin(TEST_MONTHS)].copy()
    month_subs = {m: test[test["month"] == m] for m in TEST_MONTHS}

    ref_y, ref_m, ref_f, ref_s = [], [], [], []
    for m in TEST_MONTHS:
        sub = month_subs[m]
        ref_y.append(sub["Label"].astype(int).values)
        ref_m.append(np.array([m] * len(sub), dtype=object))
        ref_f.append(sub["family"].astype(str).values)
        ref_s.append(sub["sha256"].astype(str).values)
    meta = {
        "y": np.concatenate(ref_y),
        "month": np.concatenate(ref_m),
        "family": np.concatenate(ref_f),
        "sha256": np.concatenate(ref_s),
    }

    preds = {}
    for spec in _model_specs():
        label = spec["label"]
        print(f"[INFO] predicting {label} ...")
        if spec["kind"] == "baseline":
            d = joblib.load(os.path.join(BASELINE_DIR, spec["file"]))
            feats = d["feat_cols"]
            probs = []
            for m in TEST_MONTHS:
                sub = month_subs[m]
                X = sub[feats].fillna(0).astype(np.float32).values
                probs.append(_predict_baseline(d["model"], d["scaler"], X))
        else:
            feats = drift_feats
            model = LatentNeuralODE(input_dim=len(feats), latent_dim=8, hidden=32,
                                    solver=spec["solver"], use_ode=spec["use_ode"])
            model.load_state_dict(torch.load(os.path.join(RESULTS, spec["file"]),
                                             map_location="cpu"))
            model.eval()
            probs = []
            for m in TEST_MONTHS:
                sub = month_subs[m]
                X = sub[feats].fillna(0).astype(np.float32).values
                t = sub["timestamp"].values.astype("datetime64[s]").astype(np.int64)
                t = (t - t.min()) / max(t.max() - t.min(), 1)
                probs.append(_predict_ode(model, X, t.astype(np.float32)))
        preds[spec["key"]] = np.concatenate(probs)

    np.savez_compressed(PRED_CACHE,
                        **{f"p_{k}": v for k, v in preds.items()}, **meta)
    print("[INFO] cached predictions ->", os.path.relpath(PRED_CACHE, ROOT))
    return preds, meta


def _metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return dict(auroc=roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
                f1_macro=f1_score(y, pred, average="macro", zero_division=0),
                recall=recall_score(y, pred, zero_division=0),
                precision=precision_score(y, pred, zero_division=0))


# ----------------------------------------------------------------------------
# 4. ROC curves + low-FPR zoom
# ----------------------------------------------------------------------------

def fig_roc(preds, meta):
    out = _out("04_roc_curves")
    y = meta["y"]
    rows = []
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 6))

    for label in DISPLAY_ORDER:
        key = next(s["key"] for s in _model_specs() if s["label"] == label)
        p = preds[key]
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        c = PALETTE[label]
        st = STYLE[label]
        ax_full.plot(fpr, tpr, color=c, label=f"{label} (AUC={auc:.4f})", **st)
        ax_zoom.plot(fpr, tpr, color=c, label=label, **st)
        # TPR at operational low-FPR budgets
        def tpr_at(budget):
            idx = np.searchsorted(fpr, budget, side="right") - 1
            return float(tpr[max(idx, 0)])
        rows.append({"model": label, "auroc": auc,
                     "tpr_at_fpr_0.001": tpr_at(0.001),
                     "tpr_at_fpr_0.01": tpr_at(0.01)})

    ax_full.plot([0, 1], [0, 1], color="#aaaaaa", ls=":", lw=1, label="Chance")
    ax_full.set_xlim(0, 1)
    ax_full.set_ylim(0, 1.01)
    ax_full.set_xlabel("False positive rate (FPR)")
    ax_full.set_ylabel("True positive rate / recall")
    ax_full.set_title("ROC curves on the chronological test window")
    ax_full.legend(loc="lower right", fontsize=8, framealpha=0.95)

    ax_zoom.set_xlim(0, 0.01)
    ax_zoom.axvline(0.01, color="#d9534f", ls="--", lw=1, alpha=0.7)
    ax_zoom.set_xlabel("False positive rate (FPR)")
    ax_zoom.set_ylabel("True positive rate / recall")
    ax_zoom.set_title("Low-FPR zoom (enterprise false-alarm budget < 1%)")
    ax_zoom.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax_zoom.text(0.0097, 0.02, "FPR = 1%", color="#d9534f", rotation=90,
                 ha="right", va="bottom", fontsize=8,
                 transform=ax_zoom.get_xaxis_transform())
    ax_zoom.text(0.03, 0.06,
                 "Neural ODE curves lie outside this region:\nthey need FPR > 1% to reach useful recall",
                 transform=ax_zoom.transAxes, fontsize=8, color="#2ca02c",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.85))

    fig.suptitle("Multi-model ROC comparison and low-FPR operating region",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _savefig(fig, os.path.join(out, "roc_curves_and_low_fpr_zoom.png"))

    # Stand-alone zoom figure
    fig2, ax = plt.subplots(figsize=(8, 6))
    for label in DISPLAY_ORDER:
        key = next(s["key"] for s in _model_specs() if s["label"] == label)
        fpr, tpr, _ = roc_curve(y, preds[key])
        ax.plot(fpr, tpr, color=PALETTE[label], label=label, **STYLE[label])
    ax.set_xlim(0, 0.01)
    ax.axvline(0.01, color="#d9534f", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate / recall")
    ax.set_title("Low-FPR zoom: detection at FPR < 1%")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
    ax.text(0.03, 0.05,
            "Neural ODE curves lie outside this region\n(FPR > 1% required)",
            transform=ax.transAxes, fontsize=8, color="#2ca02c",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.85))
    _savefig(fig2, os.path.join(out, "low_fpr_zoom.png"))

    pd.DataFrame(rows).to_csv(os.path.join(out, "roc_summary.csv"), index=False)
    return rows


# ----------------------------------------------------------------------------
# 5. Monthly macro-F1 / recall decay
# ----------------------------------------------------------------------------

def fig_f1_recall(preds, meta):
    out = _out("05_f1_recall_decay")
    y, month = meta["y"], meta["month"]
    rows = []
    for spec in _model_specs():
        p = preds[spec["key"]]
        for m in TEST_MONTHS:
            mask = month == m
            mm = _metrics(y[mask], p[mask])
            rows.append(dict(model=spec["label"], month=m,
                             support=int(mask.sum()),
                             malware=int((y[mask] == 1).sum()),
                             benign=int((y[mask] == 0).sum()), **mm))
        mm = _metrics(y, p)
        rows.append(dict(model=spec["label"], month="OVERALL",
                         support=int(len(y)), malware=int((y == 1).sum()),
                         benign=int((y == 0).sum()), **mm))
    tbl = pd.DataFrame(rows)
    tbl.to_csv(os.path.join(out, "monthly_f1_recall.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True)
    for label in DISPLAY_ORDER:
        sub = tbl[(tbl["model"] == label) & (tbl["month"] != "OVERALL")]
        x = np.arange(len(TEST_MONTHS))
        axes[0].plot(x, sub["f1_macro"].values, marker="o", markersize=6,
                     color=PALETTE[label], label=label, **STYLE[label])
        axes[1].plot(x, sub["recall"].values, marker="o", markersize=6,
                     color=PALETTE[label], label=label, **STYLE[label])
    for ax, title, col in [(axes[0], "Monthly macro-F1 (threshold p >= 0.5)", "f1_macro"),
                           (axes[1], "Monthly malware recall (threshold p >= 0.5)", "recall")]:
        ax.set_xticks(np.arange(len(TEST_MONTHS)))
        ax.set_xticklabels([pd.Period(m).strftime("%b %Y") for m in TEST_MONTHS])
        ax.set_xlabel("Chronological test month")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.3, linestyle=":")
    axes[0].legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.suptitle("Operational stability at a fixed decision threshold",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    _savefig(fig, os.path.join(out, "monthly_f1_recall_decay.png"))
    return tbl


# ----------------------------------------------------------------------------
# 6. Confusion matrices
# ----------------------------------------------------------------------------

def fig_confusion(preds, meta):
    out = _out("06_confusion_matrices")
    y = meta["y"]
    rows = []
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    specs = _model_specs()
    ax_list = list(axes.ravel())
    for ax in ax_list[len(specs):]:
        ax.axis("off")
    for ax, spec in zip(ax_list, specs):
        p = preds[spec["key"]]
        pred = (p >= 0.5).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        m = _metrics(y, p)
        rows.append(dict(model=spec["label"], TN=int(tn), FP=int(fp),
                         FN=int(fn), TP=int(tp), **m))
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        for i in range(2):
            for j in range(2):
                v = cm[i, j]
                frac = v / cm.sum()
                color = "white" if v > cm.max() * 0.55 else "#222222"
                ax.text(j, i, f"{v:,}\n({frac:.2%})", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred benign", "Pred malware"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True benign", "True malware"])
        ax.set_title(f"{spec['label']}\nRecall={m['recall']:.3f}  Precision={m['precision']:.3f}  "
                     f"F1={m['f1_macro']:.3f}", fontsize=10)
        ax.grid(False)
    fig.suptitle("Confusion matrices on the full test window (38,001 samples)",
                 fontsize=14, y=1.0)
    plt.tight_layout()
    _savefig(fig, os.path.join(out, "confusion_matrix_comparison.png"))
    pd.DataFrame(rows).to_csv(os.path.join(out, "confusion_matrix_counts.csv"),
                              index=False)
    return rows


# ----------------------------------------------------------------------------
# 7. Architecture + workflow diagrams
# ----------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, fc, ec="#333333", fs=9, bold=True):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       linewidth=1.2, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, x1, y1, x2, y2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                        linewidth=1.4, color="#444444", zorder=1)
    ax.add_patch(a)


def fig_architecture():
    out = _out("07_system_architecture")

    fig, ax = plt.subplots(figsize=(18, 4.8))
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    stages = [
        ("PE feature vector\nx ∈ R^600\n(F1 … F2384, drift-ranked)", "#e9ecef"),
        ("Encoder\nMLP\n600 → 32 → 8", "#cfe2f3"),
        ("Initial latent\nz(t₀) ∈ R^8", "#d9d2e9"),
        ("Latent ODE\n dz/dt = f(z, t)\nintegrate t: 0 → T\n(dopri5 / rk4)", "#fce5cd"),
        ("Evolved latent\nz(T) ∈ R^8", "#d9d2e9"),
        ("Decoder\n[z(T); t] → 32 → 1", "#d9ead3"),
        ("Prediction\np(malware) = σ(·)", "#f4cccc"),
    ]
    w, h, gap, y0 = 2.2, 1.6, 0.5, 1.9
    x = 0.2
    centers = []
    for text, color in stages:
        _box(ax, x, y0, w, h, text, color, fs=10)
        centers.append((x, x + w))
        x += w + gap
    for i in range(len(stages) - 1):
        _arrow(ax, centers[i][1] + 0.02, y0 + h / 2,
               centers[i + 1][0] - 0.02, y0 + h / 2)

    ax.text(18.2, 0.7,
            "Loss: Focal Loss (α=0.25, γ=2.0)   |   Optimizer: Adam (lr=1e-3)   |   "
            "Backprop: adjoint sensitivity (constant memory)",
            ha="right", va="center", fontsize=10, color="#333333")
    ax.text(0.2, 4.45, "Fig. 1 — Latent Neural ODE system architecture",
            ha="left", va="center", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, os.path.join(out, "fig1_system_architecture.png"))

    fig, ax = plt.subplots(figsize=(18, 5.6))
    ax.set_xlim(0, 18.4)
    ax.set_ylim(0, 5.6)
    ax.axis("off")
    boxes = [
        ("1. BODMAS dataset\n\n121,870 PE samples\nF1–F2381 features\nLabel + timestamp + family",
         "#e9ecef"),
        ("2. Chronological split\n\nTrain  Aug 2019 – Mar 2020   62,919\nVal    Apr – May 2020        20,950\nTest   Jun – Sep 2020        38,001",
         "#cfe2f3"),
        ("3. Model training\n\nStatic: XGBoost, LightGBM, MLP\nContinuous: Latent Neural ODE\n(dopri5 / rk4) + decoupled ablation",
         "#fce5cd"),
        ("4. Evaluation\n\nPer-month AUROC, macro-F1, recall\nAUT, ROC / low-FPR, confusion\nNovel vs persistent families",
         "#d9ead3"),
    ]
    w, h, gap, y0 = 4.0, 3.2, 0.6, 1.3
    x = 0.3
    spans = []
    for text, color in boxes:
        _box(ax, x, y0, w, h, text, color, fs=10.5, bold=False)
        spans.append((x, x + w))
        x += w + gap
    for i in range(len(boxes) - 1):
        _arrow(ax, spans[i][1] + 0.02, y0 + h / 2,
               spans[i + 1][0] - 0.02, y0 + h / 2)
    ax.text(0.3, 5.25, "Fig. 2 — Experimental workflow",
            ha="left", va="center", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _savefig(fig, os.path.join(out, "fig2_experimental_workflow.png"))


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Generating deliverable graphs from real data")
    print("=" * 70)

    df, baseline_feats, drift_feats = load_data()
    print(f"[INFO] loaded {len(df):,} rows | baseline feats={len(baseline_feats)} "
          f"| ODE feats={len(drift_feats)}")

    fig_temporal_split(df)
    fig_top_drift()

    if os.path.exists(NOVEL_TXT):
        novel = _parse_novel_txt(NOVEL_TXT)
        fig_novel(novel)
    else:
        print("[WARN] novel_family_analysis.txt not found; skipping fig 3")

    preds, meta = build_predictions(df, baseline_feats, drift_feats)
    fig_roc(preds, meta)
    fig_f1_recall(preds, meta)
    fig_confusion(preds, meta)
    fig_architecture()

    print("\n[INFO] All figures written under", os.path.relpath(GRAPH_DIR, ROOT))


if __name__ == "__main__":
    main()
