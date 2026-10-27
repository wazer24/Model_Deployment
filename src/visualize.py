"""
Visualizations built from REAL computed metrics (no synthetic data).

Produces:
  - graphs/auroc_vs_month.png     AUROC month-by-month for every model
  - graphs/aut_comparison.png     AUT bar chart + cumulative AUT curves
  - graphs/umap_latent_space.png  initial (z0) vs evolved (z_t) latent states
                                    from the trained Neural ODE dopri5 model

Reads metrics from results/*_metrics.csv and the saved ODE checkpoint.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from chronology import load_dataframe, split_chronological, get_arrays, normalize_time
from evaluate import _normalize_ts
from ode_model import LatentNeuralODE

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)

DEVICE = "cpu"
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# Friendly model -> short label for the plot legend
LABEL_MAP = {
    "XGBoost (static)": "XGBoost (static)",
    "LightGBM (static)": "LightGBM (static)",
    "MLP (static)": "MLP (static)",
    "Neural ODE (dopri5, focal)": "Neural ODE (dopri5)",
    "Neural ODE (rk4, focal)": "Neural ODE (rk4)",
    "Decoupled encoder/decoder (no ODE)": "Decoupled (no ODE)",
}

PALETTE = {
    "XGBoost (static)": "#d62728",
    "LightGBM (static)": "#ff7f0e",
    "MLP (static)": "#9467bd",
    "Neural ODE (dopri5, focal)": "#2ca02c",
    "Neural ODE (rk4, focal)": "#1f77b4",
    "Decoupled encoder/decoder (no ODE)": "#7f7f7f",
}
STATIC = {"XGBoost (static)", "LightGBM (static)", "MLP (static)"}


def load_all_metrics():
    frames = []
    paths = sorted(glob.glob(os.path.join(RESULTS_DIR, "*metrics*.csv")))
    paths = [p for p in paths if "comparison_table" not in os.path.basename(p)]
    for path in paths:
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError("No *_metrics.csv found in results/")
    allm = pd.concat(frames, ignore_index=True)
    return allm


def month_label(m):
    if m == "OVERALL":
        return "Overall"
    p = pd.Period(m)
    return p.strftime("%b %Y")


def plot_auroc_vs_month(metrics_df):
    test_months = [m for m in metrics_df["month"].unique() if m != "OVERALL"]
    test_months = sorted(test_months)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name in metrics_df["model"].unique():
        sub = metrics_df[metrics_df["model"] == model_name].set_index("month").loc[test_months]
        if sub.empty:
            continue
        color = PALETTE.get(model_name, None)
        is_static = model_name in STATIC
        ax.plot(
            range(len(test_months)),
            sub["auroc"].values,
            marker="o", markersize=6, linewidth=2.2,
            label=LABEL_MAP.get(model_name, model_name),
            color=color,
            linestyle="--" if is_static else "-",
            alpha=0.9 if is_static else 1.0,
        )

    ax.set_xlabel("Chronological test month", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("Month-by-Month AUROC (real evaluated metrics)", fontsize=13, pad=14)
    ax.set_xticks(range(len(test_months)))
    ax.set_xticklabels([month_label(m) for m in test_months], rotation=0)
    ax.set_ylim(0.4, 1.02)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(GRAPH_DIR, "auroc_vs_month.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved", out)
    return out


def plot_aut_comparison(metrics_df, test_months):
    overall = metrics_df[metrics_df["month"] == "OVERALL"].set_index("model")
    models = [m for m in overall.index if m in LABEL_MAP]
    aut_vals = [float(overall.loc[m, "aut"]) if not np.isnan(overall.loc[m, "aut"]) else 0.0 for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1, 1.35]})

    # Left: AUT bar chart
    ax = axes[0]
    colors = [PALETTE.get(m, "#333333") for m in models]
    bars = ax.bar(range(len(models)), aut_vals, color=colors, width=0.6, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([LABEL_MAP[m] for m in models], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AUT (area under AUROC-over-time)", fontsize=11)
    ax.set_title("AUT per model (higher = more drift-resilient)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    for b, v in zip(bars, aut_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    # Right: cumulative AUT curve month-by-month
    ax = axes[1]
    for model_name in metrics_df["model"].unique():
        sub = metrics_df[(metrics_df["model"] == model_name) & (metrics_df["month"] != "OVERALL")].set_index("month").loc[test_months]
        if sub.empty:
            continue
        aurocs = sub["auroc"].values.astype(float)
        cum = np.cumsum(aurocs)  # running area proxy
        color = PALETTE.get(model_name, None)
        is_static = model_name in STATIC
        ax.plot(
            range(len(test_months)), cum,
            marker="o", markersize=5, linewidth=2,
            label=LABEL_MAP.get(model_name, model_name),
            color=color, linestyle="--" if is_static else "-",
        )
    ax.set_xlabel("Chronological test month", fontsize=11)
    ax.set_ylabel("Cumulative AUROC (AUT proxy)", fontsize=11)
    ax.set_title("Cumulative AUROC over test window", fontsize=12)
    ax.set_xticks(range(len(test_months)))
    ax.set_xticklabels([month_label(m) for m in test_months], fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left")

    fig.suptitle("Area Under Time (AUT): static baselines vs continuous-time models", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(GRAPH_DIR, "aut_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved", out)
    return out


def plot_umap_latent(model_path=None, n_samples=900):
    """UMAP of initial (z0) vs evolved (z_t) latent states from the trained ODE."""
    try:
        import umap
        HAS_UMAP = True
    except Exception:
        HAS_UMAP = False
        from sklearn.decomposition import PCA

    df, feat_cols = load_dataframe(max_features=600)
    split_info = split_chronological(df)

    # gather a sample across all test months
    X_parts, y_parts, t_parts = [], [], []
    rng = np.random.RandomState(RANDOM_STATE)
    for m in split_info["test_months"]:
        idx = split_info["month_idx"][m]
        Xm = df.loc[idx, feat_cols].fillna(0).astype(np.float32).values
        ym = df.loc[idx, "Label"].astype(np.float32).values
        tm = normalize_time(df.loc[idx, "timestamp"].values)
        n = min(n_samples // len(split_info["test_months"]), Xm.shape[0])
        sel = rng.choice(Xm.shape[0], size=n, replace=False)
        X_parts.append(Xm[sel]); y_parts.append(ym[sel]); t_parts.append(tm[sel])
    X = np.vstack(X_parts); y = np.concatenate(y_parts); t = np.concatenate(t_parts)

    model = LatentNeuralODE(input_dim=X.shape[1], latent_dim=8, hidden=32,
                            solver="dopri5", use_ode=True).to(DEVICE)
    ckpt = model_path or os.path.join(RESULTS_DIR, "ode_model_dopri5_focal.pt")
    if not os.path.exists(ckpt):
        print("[WARN] ODE checkpoint not found; using synthetic latent UMAP.")
        _umap_synthetic()
        return
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        Xb = torch.from_numpy(X).to(DEVICE)
        tb = torch.from_numpy(t).to(DEVICE)
        _, z0, zt = model(Xb, tb)
    z0 = z0.cpu().numpy(); zt = zt.cpu().numpy()

    both = np.vstack([z0, zt])
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)
        Z = reducer.fit_transform(both)
    else:
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=42).fit_transform(both)
    n = z0.shape[0]
    Z0, Zt = Z[:n], Z[n:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {0: "#1f77b4", 1: "#d62728"}
    labels_map = {0: "Benign", 1: "Malware"}

    for ax, Zz, title in [(axes[0], Z0, "Latent — initial z(t₀)"),
                          (axes[1], Zt, "Latent — evolved z(t₁)")]:
        for cls in np.unique(y):
            mask = y == cls
            ax.scatter(Zz[mask, 0], Zz[mask, 1], c=colors[cls], label=labels_map[cls],
                       s=22, alpha=0.7, edgecolor="k", linewidth=0.25)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Component 1", fontsize=10)
        ax.set_ylabel("Component 2", fontsize=10)
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.25, linestyle=":")

    fig.suptitle("Latent Space Migration: initial vs evolved states (trained Neural ODE)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(GRAPH_DIR, "umap_latent_space.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("[INFO] Saved", out)


def _umap_synthetic():
    # fallback not used unless checkpoint missing
    pass


def write_comparison_table(metrics_df):
    overall = metrics_df[metrics_df["month"] == "OVERALL"].copy()
    overall = overall.sort_values("auroc", ascending=False)
    rows = []
    for _, r in overall.iterrows():
        rows.append({
            "model": LABEL_MAP.get(r["model"], r["model"]),
            "type": "static" if r["model"] in STATIC else "continuous",
            "auroc": round(float(r["auroc"]), 4),
            "macro_f1": round(float(r["f1_macro"]), 4),
            "aut": round(float(r["aut"]), 4) if not np.isnan(r["aut"]) else None,
        })
    tbl = pd.DataFrame(rows)
    out = os.path.join(RESULTS_DIR, "comparison_table.csv")
    tbl.to_csv(out, index=False)
    print(tbl.to_string(index=False))
    print("[INFO] Saved", out)
    return tbl


def main():
    metrics_df = load_all_metrics()
    test_months = sorted([m for m in metrics_df["month"].unique() if m != "OVERALL"])

    print("=" * 60)
    print("COMPARISON TABLE (real metrics)")
    print("=" * 60)
    write_comparison_table(metrics_df)

    plot_auroc_vs_month(metrics_df)
    plot_aut_comparison(metrics_df, test_months)
    plot_umap_latent()
    print("\n[INFO] All graphs saved to", GRAPH_DIR)


if __name__ == "__main__":
    main()
