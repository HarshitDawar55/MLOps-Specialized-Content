from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

CONTRACT_LEVELS = ["month-to-month", "one-year", "two-year"]
PAYMENT_LEVELS = ["card", "bank-transfer", "electronic-check", "mailed-check"]

# The maximum fraction of nulls tolerated in a required numeric column.
NULL_TOLERANCE = 0.02


def _null_rate_under(limit: float):
    """Wide check: fails when too MANY rows are null, not on each null row.

    ignore_na=False is essential and easy to miss. Pandera drops nulls before
    running checks by default, so a null-rate check written without it silently
    never fires -- the exact failure mode this gate exists to catch.
    """
    return Check(
        lambda s: s.isna().mean() <= limit,
        element_wise=False,
        ignore_na=False,
        error=f"null_rate_exceeds_{limit}",
    )


training_schema = DataFrameSchema(
    {
        "customer_id": Column(str, nullable=False, unique=True),
        "tenure_months": Column(
            int,
            checks=[Check.in_range(0, 120, error="tenure_out_of_range_0_120")],
            nullable=False,
        ),
        "monthly_charges": Column(
            float,
            checks=[
                Check.in_range(0, 500, error="monthly_charges_out_of_range"),
                _null_rate_under(NULL_TOLERANCE),
            ],
            nullable=True,  # nullable, but the null RATE is capped above
        ),
        "total_charges": Column(
            float,
            checks=[Check.ge(0, error="total_charges_negative")],
            nullable=False,
        ),
        "support_tickets": Column(
            int,
            checks=[Check.in_range(0, 50, error="support_tickets_out_of_range")],
            nullable=False,
        ),
        "avg_latency_ms": Column(
            float,
            checks=[Check.in_range(0, 2000, error="latency_out_of_range")],
            nullable=False,
            coerce=False,  # a string here must FAIL, not be silently coerced
        ),
        "contract_type": Column(
            str,
            checks=[Check.isin(CONTRACT_LEVELS, error="unseen_contract_type")],
            nullable=False,
        ),
        "payment_method": Column(
            str,
            checks=[Check.isin(PAYMENT_LEVELS, error="unseen_payment_method")],
            nullable=False,
        ),
        "has_fiber": Column(
            int,
            checks=[Check.isin([0, 1], error="has_fiber_not_binary")],
            nullable=False,
        ),
        "churned": Column(
            int, checks=[Check.isin([0, 1], error="target_not_binary")], nullable=False
        ),
    },
    strict=False,  # extra columns allowed (batch/severity in the drift file)
    coerce=False,  # do not paper over type problems
    name="churn_training_contract",
)

# Serving sees the same columns minus the label.
serving_schema = DataFrameSchema(
    {k: v for k, v in training_schema.columns.items() if k != "churned"},
    strict=False,
    coerce=False,
    name="churn_serving_contract",
)


class ValidationFailed(Exception):
    """Raised by validate() so a pipeline can halt deliberately."""

    def __init__(self, report: pd.DataFrame):
        self.report = report
        super().__init__(f"{len(report)} validation failure(s)")


def validate(df: pd.DataFrame, schema: DataFrameSchema | None = None) -> pd.DataFrame:
    """Validate a batch. Returns the frame, or raises ValidationFailed.

    lazy=True collects EVERY violation instead of stopping at the first --
    which is what makes the on-screen report useful rather than a single line.
    """
    schema = schema or training_schema
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raise ValidationFailed(exc.failure_cases) from exc


def summarize_failures(failure_cases: pd.DataFrame) -> pd.DataFrame:
    """Collapse row-level failures into a readable per-check summary."""
    cols = [
        c for c in ("column", "check", "failure_case") if c in failure_cases.columns
    ]
    grouped = (
        failure_cases.groupby(cols[:2], dropna=False)
        .size()
        .reset_index(name="failing_rows")
        .sort_values("failing_rows", ascending=False)
    )
    return grouped
