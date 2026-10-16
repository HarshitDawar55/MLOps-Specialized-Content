import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import demo_env 

demo_env.quiet_blas_fp_warnings()

import mlflow_setup  
import modeling 

DATA = ROOT / "data"
WORKDIR = ROOT / "artifacts" / "session01"
EXPERIMENT = "session01-reproducibility"


def banner(text: str) -> None:
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def tracked_run(run_name: str, source_csv: Path, live_path: Path) -> dict:
    """Train once, recording everything needed to rebuild this exact model."""
    shutil.copy(source_csv, live_path)
    df = pd.read_csv(live_path)

    prov = modeling.provenance(live_path, df)

    with mlflow.start_run(run_name=run_name) as run:
        model, metrics = modeling.train_and_eval(df)

        # Parameters: the knobs a human chose.
        mlflow.log_params(
            {
                "model_kind": "logreg",
                "class_weight": "balanced",
                "test_size": 0.25,
                "random_state": modeling.RANDOM_STATE,
            }
        )

        # Metrics: what came out.
        mlflow.log_metrics(metrics)

        # Tags: the provenance. This is the part people skip and later regret.
        mlflow.set_tags(
            {
                "data_file": live_path.name,
                "data_file_sha256": prov["data_file_sha256"],
                "data_content_sha256": prov["data_content_sha256"],
                "data_rows": prov["data_rows"],
                "git_commit": prov["git_commit"],
                "python": prov["environment"]["python"],
                "numpy": prov["environment"]["numpy"],
                "pandas": prov["environment"]["pandas"],
                "scikit_learn": prov["environment"]["scikit_learn"],
            }
        )

        # The data snapshot itself, so the run is self-contained.
        mlflow.log_artifact(str(live_path), artifact_path="input_data")

        print(f"\n  run_name   : {run_name}")
        print(f"  run_id     : {run.info.run_id}")
        print(f"  ROC-AUC    : {metrics['roc_auc']}")
        print(f"  data sha256: {prov['data_content_sha256'][:32]}...")
        print(f"  git commit : {prov['git_commit']}")

        return {"run_id": run.info.run_id, "metrics": metrics, "provenance": prov}


def main() -> None:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    mlflow_setup.init(EXPERIMENT)

    live = WORKDIR / "customer_data.csv"

    banner("STEP 2  The same two runs, now recorded")

    monday = tracked_run("monday", DATA / "churn_v1.csv", live)
    thursday = tracked_run("thursday", DATA / "churn_v2.csv", live)

    banner("Now the question is answerable")

    same_code = (
        monday["provenance"]["git_commit"] == thursday["provenance"]["git_commit"]
    )
    same_env = (
        monday["provenance"]["environment"] == thursday["provenance"]["environment"]
    )
    same_data = (
        monday["provenance"]["data_content_sha256"]
        == thursday["provenance"]["data_content_sha256"]
    )

    print(f"\n  code identical        : {same_code}")
    print(f"  environment identical : {same_env}")
    print(f"  data identical        : {same_data}   <-- the culprit")

    print(
        f"\n  AUC moved {monday['metrics']['roc_auc']} -> {thursday['metrics']['roc_auc']}"
    )
    print("  Two of the three legs are provably unchanged, so the data moved.")
    print("  That is a five-second diagnosis instead of a two-day argument.")

    mlflow_setup.print_ui_hint()


if __name__ == "__main__":
    main()
