from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
import pandas as pd  # noqa: E402

import features  # noqa: E402
import modeling  # noqa: E402
import schema as data_schema  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
import demo_env  # noqa: E402

demo_env.quiet_blas_fp_warnings()  # spurious BLAS FP warnings; see scripts/diagnose_env.py

REGISTERED_MODEL = "churn-classifier"


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: str = ""
    payload: Any = None
    metrics: dict = field(default_factory=dict)


class PipelineHalted(Exception):
    """Raised when a gate stops the pipeline deliberately."""

    def __init__(self, stage: str, detail: str, report: pd.DataFrame | None = None):
        self.stage = stage
        self.detail = detail
        self.report = report
        super().__init__(f"halted at stage '{stage}': {detail}")


# ---------------------------------------------------------------------------
# Stage 1 - ingest
# ---------------------------------------------------------------------------
def ingest(data_path: str | Path) -> tuple[pd.DataFrame, dict]:
    path = Path(data_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise PipelineHalted("ingest", f"file not found: {path}")

    df = pd.read_csv(path)
    prov = modeling.provenance(path, df)
    return df, prov


# ---------------------------------------------------------------------------
# Stage 2 - validate  (THE GATE)
# ---------------------------------------------------------------------------
def validate(df: pd.DataFrame) -> StageResult:
    """Halt before any compute is spent on a bad batch.

    The business argument for this stage: training on a corrupt batch does not
    merely waste GPU hours, it produces a model that looks fine by its own
    metrics and is wrong in production. Cheaper to stop here.
    """
    try:
        data_schema.validate(df)
        return StageResult("validate", True, f"{len(df):,} rows passed the contract")
    except data_schema.ValidationFailed as exc:
        summary = data_schema.summarize_failures(exc.report)
        raise PipelineHalted(
            "validate",
            f"{len(summary)} distinct check(s) failed across {int(summary['failing_rows'].sum())} rows",
            report=summary,
        ) from exc


# ---------------------------------------------------------------------------
# Stage 3 - featurize
# ---------------------------------------------------------------------------
def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Separate from training on purpose -- serving calls this same function."""
    return features.build_matrix(df)


# ---------------------------------------------------------------------------
# Stage 4 + 5 - train and evaluate
# ---------------------------------------------------------------------------
def train(df: pd.DataFrame, kind: str = "logreg", **overrides):
    X_tr, X_te, y_tr, y_te = modeling.split(df)
    model = modeling.make_model(kind, **overrides)
    model.fit(X_tr, y_tr)
    return model, (X_te, y_te)


def evaluate(model, holdout) -> dict:
    X_te, y_te = holdout
    return modeling.evaluate(model, X_te, y_te)


# ---------------------------------------------------------------------------
# Stage 6 - register
# ---------------------------------------------------------------------------
def register(
    model,
    metrics: dict,
    provenance: dict,
    kind: str,
    sample_input: pd.DataFrame,
    registered_name: str = REGISTERED_MODEL,
    extra_tags: dict | None = None,
) -> dict:
    """Log the run AND register a model version carrying full lineage.

    A model in the registry without lineage is a liability: you can deploy it
    but you cannot explain it. Every tag here answers a question an auditor,
    or your own on-call self at 2am, will eventually ask.
    """
    signature = mlflow.models.infer_signature(
        sample_input, model.predict_proba(sample_input)[:, 1]
    )

    with mlflow.start_run(run_name=f"train-{kind}") as run:
        mlflow.log_params(
            {
                "model_kind": kind,
                "test_size": 0.25,
                "random_state": modeling.RANDOM_STATE,
            }
        )
        mlflow.log_metrics(metrics)

        tags = {
            "data_file": Path(provenance["data_path"]).name,
            "data_content_sha256": provenance["data_content_sha256"],
            "data_rows": provenance["data_rows"],
            "git_commit": provenance["git_commit"],
            "python": provenance["environment"]["python"],
            "scikit_learn": provenance["environment"]["scikit_learn"],
            "feature_columns": json.dumps(features.feature_columns()),
        }
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)

        # MLflow 3.x serialises sklearn models with skops and REFUSES tree
        # ensembles unless you explicitly trust the node-storage type. This is
        # a deliberate supply-chain control: a tampered tree file can carry
        # out-of-range node indices that segfault the process on predict().
        # Trust only the specific type you have reviewed -- never dump in
        # everything get_untrusted_types() reports just to make it load.
        trusted = ["sklearn.tree._tree.Tree"] if kind == "forest" else None

        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=sample_input.head(3),
            registered_model_name=registered_name,
            skops_trusted_types=trusted,
        )

        return {
            "run_id": run.info.run_id,
            "model_uri": info.model_uri,
            "registered_name": registered_name,
        }


def latest_version(registered_name: str = REGISTERED_MODEL) -> str | None:
    client = mlflow.MlflowClient()
    versions = client.search_model_versions(f"name='{registered_name}'")
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))
