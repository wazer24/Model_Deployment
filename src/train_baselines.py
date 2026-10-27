"""
Static baseline training + month-by-month evaluation.

Models:
  - XGBoost (static)
  - LightGBM (static)
  - MLP (parameter-matched static baseline; sklearn MLPClassifier)

All are trained once on the chronological train split (2019-08..2020-03)
and evaluated month-by-month on the four test months (Jun-Sep 2020) to
measure AUROC decay under concept drift.

Outputs:
  - results/baselines_metrics.csv   (per-month + overall AUROC, F1, AUT)
  - results/baseline_models/        (pickled XGBoost / LightGBM / MLP)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from chronology import load_dataframe, split_chronological, get_arrays, normalize_time
from evaluate import evaluate_model, _normalize_ts

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:  # pragma: no cover
    HAS_LGBM = False

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
MODELS_DIR = os.path.join(RESULTS_DIR, "baseline_models")
METRICS_CSV = os.path.join(RESULTS_DIR, "baselines_metrics.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42


def make_baselines():
    models = {}
    if HAS_XGB:
        models["XGBoost (static)"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
    if HAS_LGBM:
        models["LightGBM (static)"] = LGBMClassifier(
            n_estimators=300, num_leaves=64, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
        )
    # parameter-matched MLP (roughly comparable capacity)
    models["MLP (static)"] = MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300,
        random_state=RANDOM_STATE, early_stopping=True,
    )
    return models


def main():
    df, feat_cols = load_dataframe(max_features=600)
    split_info = split_chronological(df)

    Xtr, ytr, _ = get_arrays(df, feat_cols, split_info["train"])

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)

    models = make_baselines()
    print(f"[INFO] Training {len(models)} static baselines on {Xtr.shape[0]} train samples, "
          f"{Xtr.shape[1]} features")

    all_rows = []
    for name, model in models.items():
        print(f"\n[INFO] Training {name} ...")
        X_fit = Xtr_s if "MLP" in name else Xtr
        model.fit(X_fit, ytr)

        # persist model (+ scaler for MLP)
        model_path = os.path.join(MODELS_DIR, name.replace(" ", "_").lower() + ".joblib")
        joblib.dump({"model": model, "scaler": scaler if "MLP" in name else None,
                     "feat_cols": feat_cols}, model_path)

        def make_predict_fn(m, use_scaler):
            def predict_fn(X, t):
                Xf = scaler.transform(X) if use_scaler else X
                prob = m.predict_proba(Xf)[:, 1]
                return prob
            return predict_fn

        res = evaluate_model(
            make_predict_fn(model, "MLP" in name),
            df, feat_cols, split_info, device="cpu",
        )
        ov = res["overall"]
        print(f"     Overall AUROC: {ov['auroc']:.4f}  Macro F1: {ov['f1_macro']:.4f}  "
              f"AUT: {ov['aut']:.4f}")
        for m in res["months"]:
            pm = res["per_month"][m]
            print(f"     {m}: AUROC={pm['auroc']:.4f} F1={pm['f1_macro']:.4f} "
                  f"(n={pm['support']}, malware={pm['malware']})")
            all_rows.append({
                "model": name, "month": m, "auroc": pm["auroc"],
                "f1_macro": pm["f1_macro"], "support": pm["support"],
                "malware": pm["malware"], "benign": pm["benign"],
            })
        all_rows.append({
            "model": name, "month": "OVERALL", "auroc": ov["auroc"],
            "f1_macro": ov["f1_macro"], "support": int((df.loc[split_info["test"], "Label"]==1).sum() +
                                                       (df.loc[split_info["test"], "Label"]==0).sum()),
            "malware": int((df.loc[split_info["test"], "Label"]==1).sum()),
            "benign": int((df.loc[split_info["test"], "Label"]==0).sum()),
            "aut": ov["aut"],
        })

    metrics_df = pd.DataFrame(all_rows)
    metrics_df.to_csv(METRICS_CSV, index=False)
    print(f"\n[INFO] Baseline metrics written to {METRICS_CSV}")
    print(metrics_df.pivot_table(index="model", columns="month", values="auroc").round(4).to_string())


if __name__ == "__main__":
    main()
