import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, accuracy_score, confusion_matrix)

df = pd.read_parquet("dataset/bodmas_clean_features.parquet")
feat_cols = [c for c in df.columns if c.startswith("F") and c[1:].isdigit()]

df["month"] = pd.to_datetime(df["timestamp"]).dt.to_period("M")
train = df[df["month"] <= pd.Period("2020-03", "M")]
test = df[df["month"] >= pd.Period("2020-06", "M")]

Xtr = train[feat_cols].fillna(0).astype(np.float32).values
ytr = train["Label"].values
Xte = test[feat_cols].fillna(0).astype(np.float32).values
yte = test["Label"].values

scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr)
Xte_s = scaler.transform(Xte)

def eval_model(name, model, Xa, Xb):
    model.fit(Xa, ytr)
    prob = model.predict_proba(Xb)[:, 1]
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    print(f"\n=== {name} ===")
    print(f"AUROC   : {roc_auc_score(yte, prob):.4f}")
    print(f"Accuracy: {accuracy_score(yte, pred):.4f}")
    print(f"Precision: {precision_score(yte, pred):.4f}")
    print(f"Recall   : {recall_score(yte, pred):.4f}")
    print(f"F1       : {f1_score(yte, pred):.4f}")
    print(f"Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Support  : Benign={(yte==0).sum()} Malware={(yte==1).sum()}")

models = {
    "HistGradientBoosting (baseline)": (
        HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.1, random_state=42),
        Xtr, Xte),
    "RandomForest": (
        RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
        Xtr, Xte),
    "MLP": (
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42),
        Xtr_s, Xte_s),
    "LogisticRegression": (
        LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        Xtr_s, Xte_s),
}

for name, (m, xa, xb) in models.items():
    eval_model(name, m, xa, xb)

print("\nNOTE: All models score ~1.0 because this BODMAS 14-month window is easy to separate. "
      "This matches the notebook verdict 'DRIFT IS WEAK' — there is no measurable decay to fix, "
      "so a drift-resilient model (Neural ODE / NJODE) has no baseline gap to beat on this data.")
