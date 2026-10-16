from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "avg_latency_ms",
    "has_fiber",
]
CATEGORICAL_FEATURES = ["contract_type", "payment_method"]
TARGET = "churned"

# Pinned category vocabulary. Serving must one-hot encode into exactly these
# columns in exactly this order, or the model receives a differently shaped
# matrix than it was trained on. Session 6's silent-failure demo exploits
# precisely this.
CONTRACT_LEVELS = ["month-to-month", "one-year", "two-year"]
PAYMENT_LEVELS = ["card", "bank-transfer", "electronic-check", "mailed-check"]

DERIVED_FEATURES = ["charges_per_month_tenure", "tickets_per_year"]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Pure function, no fitting, no state."""
    out = df.copy()
    # Guard against divide-by-zero on tenure of 0.
    safe_tenure = out["tenure_months"].clip(lower=1)
    out["charges_per_month_tenure"] = (out["total_charges"] / safe_tenure).round(4)
    out["tickets_per_year"] = (out["support_tickets"] / safe_tenure * 12).round(4)
    return out


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode against the PINNED vocabulary, not whatever is present.

    Using pd.get_dummies on the batch alone is the classic skew bug: a batch
    missing one category produces fewer columns and the model silently
    receives misaligned input.
    """
    out = df.copy()
    for level in CONTRACT_LEVELS:
        out[f"contract_{level}"] = (out["contract_type"] == level).astype(int)
    for level in PAYMENT_LEVELS:
        out[f"payment_{level}"] = (out["payment_method"] == level).astype(int)
    return out


def feature_columns() -> list[str]:
    """The exact column list, in order, that the model expects."""
    return (
        NUMERIC_FEATURES
        + DERIVED_FEATURES
        + [f"contract_{c}" for c in CONTRACT_LEVELS]
        + [f"payment_{p}" for p in PAYMENT_LEVELS]
    )


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature path: engineer -> encode -> select, in fixed order.

    Everything is cast to float64 on the way out. This is not cosmetic:
    integer columns cannot represent missing values, so a model whose schema
    declares ints will reject an inference payload that contains a null --
    at runtime, in production, on a Friday. Declaring doubles up front is the
    documented way to avoid it.
    """
    return encode(engineer(df))[feature_columns()].astype("float64")


def build_matrix_skewed(df: pd.DataFrame) -> pd.DataFrame:
    """A BROKEN variant used only by Session 6's silent-failure demo.

    THE BUG: an upstream service starts reporting latency in SECONDS while
    the model was trained on MILLISECONDS. A one-line unit mismatch.

    Why this is the right bug to demonstrate:
      - the output has the correct shape, so nothing raises
      - every value is within the model's trained range, so no validator trips
      - the model returns confident probabilities, so no confidence check fires
      - the service stays at 200 / flat latency / zero errors
      - but the positive rate collapses from ~0.40 to ~0.10, meaning the
        business silently stops catching three quarters of its churners

    That is the expensive shape of failure: not an outage, an under-reaction
    that nobody sees until the quarterly numbers come in. Never ship this;
    it exists to be caught.
    """
    broken = df.copy()
    broken["avg_latency_ms"] = broken["avg_latency_ms"] / 1000.0  # ms -> s
    return encode(engineer(broken))[feature_columns()].astype("float64")


def summary(df: pd.DataFrame) -> dict:
    """Small profile used for logging and drift baselines."""
    mat = build_matrix(df)
    return {
        "rows": int(len(mat)),
        "columns": int(mat.shape[1]),
        "means": {c: float(np.round(mat[c].mean(), 4)) for c in mat.columns},
    }
