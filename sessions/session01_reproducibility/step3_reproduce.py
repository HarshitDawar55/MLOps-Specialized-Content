from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep the screen clean during a live demo: no tqdm download bars.
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  
import pandas as pd  

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import demo_env  
demo_env.quiet_blas_fp_warnings()

import mlflow_setup  
import modeling 

EXPERIMENT = "session01-reproducibility"


def banner(text: str) -> None:
    print()
    print("=" * 74)
    print(f"  {text}")
    print("=" * 74)


def load_runs() -> pd.DataFrame:
    exp = mlflow.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print(f"\n  No experiment '{EXPERIMENT}' found.")
        print("  Run step2_tracked_training.py first.\n")
        sys.exit(1)
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    if runs.empty:
        print("\n  No runs recorded yet. Run step2_tracked_training.py first.\n")
        sys.exit(1)
    return runs


def list_runs() -> None:
    runs = load_runs()
    banner("Recorded runs")
    cols = {
        "tags.mlflow.runName": "name",
        "run_id": "run_id",
        "metrics.roc_auc": "roc_auc",
        "tags.data_content_sha256": "data_sha256",
    }
    view = runs[[c for c in cols if c in runs.columns]].rename(columns=cols)
    if "data_sha256" in view.columns:
        view["data_sha256"] = view["data_sha256"].str.slice(0, 16) + "..."
    print()
    print(view.to_string(index=False))
    print()


def pick_run(run_name: str | None, run_id: str | None) -> pd.Series:
    runs = load_runs()
    if run_id:
        match = runs[runs["run_id"] == run_id]
    elif run_name:
        match = runs[runs.get("tags.mlflow.runName") == run_name]
    else:
        match = runs.head(1)
    if match.empty:
        print(f"\n  No run matching name={run_name!r} id={run_id!r}\n")
        sys.exit(1)
    return match.iloc[0]


def compare(label: str, recorded, current) -> bool:
    ok = str(recorded) == str(current)
    mark = "MATCH   " if ok else "DIVERGED"
    print(f"    [{mark}] {label}")
    if not ok:
        print(f"                recorded : {recorded}")
        print(f"                current  : {current}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce and diagnose an MLflow run")
    parser.add_argument("--list", action="store_true", help="list recorded runs")
    parser.add_argument("--run-name", help="run name to reproduce (e.g. monday)")
    parser.add_argument("--run-id", help="run id to reproduce")
    parser.add_argument(
        "--data",
        help="CSV to reproduce against. Defaults to the snapshot logged with the run.",
    )
    args = parser.parse_args()

    mlflow_setup.init(EXPERIMENT)

    if args.list:
        list_runs()
        return

    run = pick_run(args.run_name, args.run_id)
    run_id = run["run_id"]
    name = run.get("tags.mlflow.runName", "(unnamed)")

    banner(f"Reproducing run '{name}'  ({run_id[:12]}...)")

    # ---- resolve the data to reproduce against -------------------------
    if args.data:
        data_path = Path(args.data)
        if not data_path.is_absolute():
            data_path = ROOT / data_path
        source = f"override: {data_path}"
    else:
        # Pull the snapshot that was logged with the run -- the whole point
        # of logging it.
        local = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="input_data"
        )
        candidates = list(Path(local).glob("*.csv"))
        if not candidates:
            print("\n  Run has no logged input_data artifact.\n")
            sys.exit(1)
        data_path = candidates[0]
        source = "logged snapshot from the run"

    print(f"\n  Reproducing against: {source}")

    df = pd.read_csv(data_path)
    prov = modeling.provenance(data_path, df)

    # ---- compare the three legs ----------------------------------------
    print("\n  Checking the three legs of reproducibility:\n")

    data_ok = compare(
        "DATA         (content sha256)",
        run.get("tags.data_content_sha256"),
        prov["data_content_sha256"],
    )
    code_ok = compare(
        "CODE         (git commit)", run.get("tags.git_commit"), prov["git_commit"]
    )
    env_ok = True
    for lib in ("python", "numpy", "pandas", "scikit_learn"):
        env_ok &= compare(
            f"ENVIRONMENT  ({lib})",
            run.get(f"tags.{lib}"),
            prov["environment"][lib],
        )

    # ---- retrain and compare metrics ------------------------------------
    _, metrics = modeling.train_and_eval(df)
    recorded_auc = float(run.get("metrics.roc_auc"))
    current_auc = metrics["roc_auc"]
    metrics_match = abs(recorded_auc - current_auc) < 1e-9

    print(f"\n  Retrained.  recorded ROC-AUC = {recorded_auc}   now = {current_auc}")

    # ---- verdict ---------------------------------------------------------
    banner("VERDICT: REPRODUCED" if metrics_match else "VERDICT: DIVERGED")

    if metrics_match:
        print("\n  Metrics match exactly. The run is reproducible.")
        print("  Data, code and environment all check out.\n")
    else:
        culprits = []
        if not data_ok:
            culprits.append("DATA")
        if not code_ok:
            culprits.append("CODE")
        if not env_ok:
            culprits.append("ENVIRONMENT")

        print(f"\n  ROC-AUC differs by {current_auc - recorded_auc:+.4f}")
        if culprits:
            print(f"  Diverged layer(s): {', '.join(culprits)}")
            print("\n  You did not have to guess. The tooling named it.")
        else:
            print("  All three legs match but metrics differ -- investigate")
            print("  nondeterminism: unset random_state, thread count, or GPU math.")
        print()


if __name__ == "__main__":
    main()
