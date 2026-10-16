import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 20260101
N_ROWS = 6000

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "avg_latency_ms",
    "contract_type",
    "payment_method",
    "has_fiber",
]
TARGET = "churned"


def _base_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    tenure = rng.integers(1, 73, size=n)
    monthly = np.round(rng.normal(70, 22, size=n).clip(18, 140), 2)
    # total_charges is intentionally derived -> gives us a realistic feature
    # engineering step and a realistic place for train/serve skew to appear.
    total = np.round(monthly * tenure * rng.uniform(0.92, 1.0, size=n), 2)
    tickets = rng.poisson(1.4, size=n)
    latency = np.round(rng.normal(48, 12, size=n).clip(5, None), 1)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n, p=[0.55, 0.27, 0.18]
    )
    payment = rng.choice(
        ["card", "bank-transfer", "electronic-check", "mailed-check"],
        size=n,
        p=[0.34, 0.26, 0.28, 0.12],
    )
    fiber = rng.choice([0, 1], size=n, p=[0.55, 0.45])

    return pd.DataFrame(
        {
            "customer_id": [f"C{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "support_tickets": tickets,
            "avg_latency_ms": latency,
            "contract_type": contract,
            "payment_method": payment,
            "has_fiber": fiber,
        }
    )


def _label(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Churn probability from a transparent linear rule plus noise.

    Written as an explicit formula so you can say on screen exactly which
    drivers the model should discover. Session 5's drift then attacks
    avg_latency_ms and support_tickets, which carry real weight here.
    """
    z = (
        -2.80
        - 0.034 * df["tenure_months"]
        + 0.014 * df["monthly_charges"]
        + 0.26 * df["support_tickets"]
        + 0.020 * df["avg_latency_ms"]
        + 0.55 * (df["contract_type"] == "month-to-month")
        - 0.42 * (df["contract_type"] == "two-year")
        + 0.38 * (df["payment_method"] == "electronic-check")
        + 0.22 * df["has_fiber"]
    )
    p = 1.0 / (1.0 + np.exp(-z))
    out = df.copy()
    out[TARGET] = (rng.uniform(size=len(df)) < p).astype(int)
    return out


def make_v1() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    return _label(_base_frame(rng, N_ROWS), rng)


def make_v2() -> pd.DataFrame:
    """Identical pipeline, one deliberately shifted feature.

    avg_latency_ms moves from ~48ms to ~63ms. Nothing else changes. Session 1
    uses this to show a model that 'reproduces' by code but not by result.
    """
    rng = np.random.default_rng(SEED)
    df = _base_frame(rng, N_ROWS)
    df["avg_latency_ms"] = np.round(
        (df["avg_latency_ms"] + rng.normal(15, 4, size=len(df))).clip(5, None), 1
    )
    return _label(df, rng)


def make_corrupt() -> pd.DataFrame:
    """A batch that must be rejected by Session 2's validation gate.

    Four distinct violations, each mapping to a different Pandera check, so the
    failure report on screen shows four separate reasons rather than one.
    """
    rng = np.random.default_rng(SEED + 7)
    df = _base_frame(rng, 1200)
    df = _label(df, rng)

    # 1. Null spike in a required numeric column (12% nulls).
    null_idx = rng.choice(df.index, size=int(0.12 * len(df)), replace=False)
    df.loc[null_idx, "monthly_charges"] = np.nan

    # 2. Out-of-range values: negative tenure is physically impossible.
    neg_idx = rng.choice(df.index, size=40, replace=False)
    df.loc[neg_idx, "tenure_months"] = -5

    # 3. Unseen category: a new contract type the model has never encountered.
    cat_idx = rng.choice(df.index, size=60, replace=False)
    df.loc[cat_idx, "contract_type"] = "weekly-rolling"

    # 4. Type violation: latency arrives as a string from an upstream change.
    df["avg_latency_ms"] = df["avg_latency_ms"].astype(object)
    str_idx = rng.choice(df.index, size=30, replace=False)
    df.loc[str_idx, "avg_latency_ms"] = "unknown"

    return df


def make_drift(n_batches: int = 6, batch_size: int = 900) -> pd.DataFrame:
    """Progressive covariate drift across batches 0..n-1.

    Batch 0 matches training. Each later batch pushes avg_latency_ms up and
    support_tickets up. Because both carry real weight in the labelling rule,
    true churn rate climbs with them -- so this is covariate drift that
    genuinely degrades the model, not cosmetic noise.
    """
    frames = []
    for b in range(n_batches):
        rng = np.random.default_rng(SEED + 100 + b)
        df = _base_frame(rng, batch_size)
        severity = b / max(n_batches - 1, 1)

        df["avg_latency_ms"] = np.round(
            (
                df["avg_latency_ms"] + severity * 34 + rng.normal(0, 3, size=len(df))
            ).clip(5, None),
            1,
        )
        df["support_tickets"] = df["support_tickets"] + rng.poisson(
            severity * 2.1, size=len(df)
        )

        df = _label(df, rng)
        df.insert(1, "batch", b)
        df.insert(2, "severity", round(severity, 3))
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    outputs = {
        "churn_v1.csv": make_v1(),
        "churn_v2.csv": make_v2(),
        "churn_corrupt.csv": make_corrupt(),
        "churn_drift.csv": make_drift(),
    }

    manifest = {}
    for name, frame in outputs.items():
        path = DATA_DIR / name
        frame.to_csv(path, index=False)
        manifest[name] = {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "sha256": _sha256(path),
        }
        print(
            f"  wrote {name:22s} rows={len(frame):6d}  sha256={manifest[name]['sha256'][:16]}..."
        )

    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    v1 = outputs["churn_v1.csv"]
    v2 = outputs["churn_v2.csv"]
    print()
    print(f"  churn rate v1          : {v1[TARGET].mean():.3f}")
    print(f"  churn rate v2          : {v2[TARGET].mean():.3f}")
    print(f"  mean latency v1        : {v1['avg_latency_ms'].mean():.1f} ms")
    print(
        f"  mean latency v2        : {v2['avg_latency_ms'].mean():.1f} ms  <- the only change"
    )
    print()
    print(f"  manifest -> {DATA_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
