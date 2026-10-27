"""
Chronological data splitting for malware concept-drift experiments.

Splits the BODMAS feature parquet into train / validation / test by calendar
month so that training data always precedes test data (no leakage):

  Train : 2019-08 .. 2020-03
  Val   : 2020-04 .. 2020-05
  Test  : 2020-06 .. 2020-09   (four consecutive test months)

Returns feature matrices (float32), labels, timestamps and per-month index
arrays so every model is evaluated on identical chronological splits.
"""

import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET_PATH = os.path.join(BASE_DIR, "dataset", "bodmas_clean_features.parquet")
DRIFT_RANKING = os.path.join(BASE_DIR, "dataset", "feature_drift_ranking.csv")

TRAIN_END = pd.Period("2020-03", "M")
VAL_START = pd.Period("2020-04", "M")
VAL_END   = pd.Period("2020-05", "M")
TEST_START = pd.Period("2020-06", "M")

FEATURE_PREFIX = "F"


def _top_feature_names(max_features):
    """Read the drift ranking and return the top max_features feature names."""
    ranking_path = DRIFT_RANKING
    if not os.path.exists(ranking_path) or max_features is None:
        return None
    rank = pd.read_csv(ranking_path)  # col0=feature name, col1=drift_magnitude
    names = rank.iloc[:, 0].astype(str).tolist()
    return names[:max_features]


def load_dataframe(path=PARQUET_PATH, max_features=None):
    """Read only metadata + the top max_features drift columns (memory bounded)."""
    top_names = _top_feature_names(max_features)
    meta_cols = ["sha256", "Label", "timestamp"]
    if top_names is not None:
        # keep only features that actually exist in the parquet
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(path)
        available = set(pf.schema_arrow.names)
        cols = [c for c in meta_cols + top_names if c in available]
    else:
        cols = None
    df = pd.read_parquet(path, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")
    if top_names is not None:
        feat_cols = [c for c in top_names if c in df.columns]
    else:
        feat_cols = [c for c in df.columns if c.startswith(FEATURE_PREFIX) and c[1:].isdigit()]
    return df, feat_cols


def split_chronological(df):
    """Return dict with train/val/test masks and per-test-month index arrays."""
    train_mask = df["month"] <= TRAIN_END
    val_mask = (df["month"] >= VAL_START) & (df["month"] <= VAL_END)
    test_mask = df["month"] >= TEST_START

    test_months = sorted(df.loc[test_mask, "month"].unique())
    month_idx = {str(m): df.index[test_mask & (df["month"] == m)].tolist() for m in test_months}

    return {
        "train": train_mask,
        "val": val_mask,
        "test": test_mask,
        "test_months": [str(m) for m in test_months],
        "month_idx": month_idx,
    }


def get_arrays(df, feat_cols, mask):
    X = df.loc[mask, feat_cols].fillna(0).astype(np.float32).values
    y = df.loc[mask, "Label"].astype(np.float32).values
    t = df.loc[mask, "timestamp"].values
    return X, y, t


def normalize_time(timestamps):
    """Map timestamps to [0, 1] fraction of the full timeline."""
    t = pd.to_datetime(timestamps)
    t0 = t.min()
    t1 = t.max()
    span = (t1 - t0).total_seconds()
    if span == 0:
        return np.zeros(len(t), dtype=np.float32)
    return np.asarray((t - t0).total_seconds() / span, dtype=np.float32)
