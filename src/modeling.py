from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import TARGET, build_matrix

RANDOM_STATE = 42


def split(df: pd.DataFrame, test_size: float = 0.25):
    X = build_matrix(df)
    y = df[TARGET].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )


def make_model(kind: str = "logreg", **overrides):
    """Return an unfitted estimator.

    logreg  -- we will use in the Session 1-3 default: fast, calibrated, explainable on screen.
    forest  -- this is the Session 3 'challenger' used to show conditional promotion.
    """
    if kind == "logreg":
        # class_weight="balanced" is a business choice, not a technical default:
        # for churn, a missed churner costs more than a wasted retention offer.
        # It also gives metrics that move visibly under drift.
        params = {
            "max_iter": 2000,
            "C": 1.0,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        params.update(overrides)
        return Pipeline(
            [("scale", StandardScaler()), ("clf", LogisticRegression(**params))]
        )
    if kind == "forest":
        params = {
            "n_estimators": 300,
            "max_depth": 9,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        }
        params.update(overrides)
        return RandomForestClassifier(**params)
    raise ValueError(f"unknown model kind: {kind!r}")


def evaluate(model, X, y) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, proba)), 4),
        "brier": round(float(brier_score_loss(y, proba)), 4),
    }


def train_and_eval(df: pd.DataFrame, kind: str = "logreg", **overrides):
    X_tr, X_te, y_tr, y_te = split(df)
    model = make_model(kind, **overrides)
    model.fit(X_tr, y_tr)
    return model, evaluate(model, X_te, y_te)



def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_frame(df: pd.DataFrame) -> str:
    """Content hash of a DataFrame, independent of file formatting."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "not-a-git-repo"
    except Exception:
        return "unavailable"


def env_fingerprint() -> dict:
    """The third leg of the reproducibility tripod: the environment."""
    import sklearn

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
    }


def provenance(data_path: str | Path, df: pd.DataFrame) -> dict:
    """Everything needed to answer 'which exact thing produced this model?'"""
    return {
        "data_path": str(data_path),
        "data_file_sha256": sha256_file(data_path),
        "data_content_sha256": sha256_frame(df),
        "data_rows": int(len(df)),
        "git_commit": git_commit(),
        "environment": env_fingerprint(),
    }


def print_provenance(prov: dict) -> None:
    print(json.dumps(prov, indent=2))
