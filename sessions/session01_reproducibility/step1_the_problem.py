import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import demo_env

demo_env.quiet_blas_fp_warnings()

import modeling

DATA = ROOT / "data"
WORKDIR = ROOT / "artifacts" / "session01"


def banner(text: str) -> None:
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def main() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    live = WORKDIR / "customer_data.csv"

    banner("STEP 1  The naive workflow: train, report a number, move on")

    # --- Monday: the data engineer delivers v1 -----------------------------
    shutil.copy(DATA / "churn_v1.csv", live)
    df = pd.read_csv(live)
    print(f"\n[Monday]  Loaded {live.name}: {len(df):,} rows")
    _, metrics_monday = modeling.train_and_eval(df)
    print(f"[Monday]  Trained model.  ROC-AUC = {metrics_monday['roc_auc']}")
    print(
        f"[Monday]  Recall = {metrics_monday['recall']}   F1 = {metrics_monday['f1']}"
    )
    print("[Monday]  Reported 'AUC 0.75' to the business. Closed the ticket.")

    # --- Thursday: upstream quietly refreshes the same filename ------------
    # Nobody told anyone. The filename did not change. The code did not change.
    shutil.copy(DATA / "churn_v2.csv", live)

    df = pd.read_csv(live)
    print(f"\n[Thursday] Loaded {live.name}: {len(df):,} rows   <- same filename")
    _, metrics_thursday = modeling.train_and_eval(df)
    print(f"[Thursday] Trained model.  ROC-AUC = {metrics_thursday['roc_auc']}")
    print(
        f"[Thursday] Recall = {metrics_thursday['recall']}   F1 = {metrics_thursday['f1']}"
    )

    banner("The result moved. Can you prove why?")

    delta = round(metrics_thursday["roc_auc"] - metrics_monday["roc_auc"], 4)
    print(f"\n  ROC-AUC moved by {delta:+}")
    print("\n  Questions you cannot currently answer:")
    print("    - Which exact rows trained Monday's model?")
    print("    - Which trained Thursday's?")
    print("    - Was it a data change, a library upgrade, or a code edit?")
    print("    - Can you rebuild Monday's model to compare them fairly?")
    print("\n  Git has your code. Git does not have your data or your environment.")
    print("  That is the gap the rest of this session closes.\n")


if __name__ == "__main__":
    main()
