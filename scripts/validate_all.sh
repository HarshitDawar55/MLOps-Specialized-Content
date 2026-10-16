set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
if [ -x "${REPO_ROOT}/../.venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/../.venv/bin/python"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

PASS=0
FAIL=0
FAILED_STEPS=()
LOG="${REPO_ROOT}/artifacts/validation.log"
mkdir -p "${REPO_ROOT}/artifacts"
: > "${LOG}"

step () {
  local label="$1"; shift
  printf "  %-58s" "${label}"
  if "$@" >> "${LOG}" 2>&1; then
    echo "PASS"
    PASS=$((PASS + 1))
  else
    echo "FAIL"
    FAIL=$((FAIL + 1))
    FAILED_STEPS+=("${label}")
  fi
}

# A step that is EXPECTED to exit non-zero (the validation gate halting).
step_expect_fail () {
  local label="$1"; shift
  printf "  %-58s" "${label}"
  if "$@" >> "${LOG}" 2>&1; then
    echo "FAIL (should have halted)"
    FAIL=$((FAIL + 1))
    FAILED_STEPS+=("${label}")
  else
    echo "PASS"
    PASS=$((PASS + 1))
  fi
}

echo ""
echo "=========================================================================="
echo "  FULL REHEARSAL"
echo "=========================================================================="
echo ""
echo "  python : $(${PYTHON} --version 2>&1)"
echo "  log    : ${LOG}"
echo ""

# --- environment -----------------------------------------------------------
echo "  ENVIRONMENT"
step "imports resolve" "${PYTHON}" -c "import mlflow,pandera,prefect,evidently,fastapi,sklearn,pandas,boto3"
step "no genuine FP errors in training" "${PYTHON}" scripts/diagnose_env.py
step "sqlalchemy is <2.1 (Prefect scheduler pin)" "${PYTHON}" -c "
import sqlalchemy,sys
major,minor=sqlalchemy.__version__.split('.')[:2]
sys.exit(0 if (int(major),int(minor))<(2,1) else 1)"
echo ""

# --- data ------------------------------------------------------------------
echo "  DATA"
step "generate datasets" "${PYTHON}" src/make_data.py
step "clean batch passes the contract" "${PYTHON}" -c "
import sys; sys.path.insert(0,'src')
import pandas as pd, schema
schema.validate(pd.read_csv('data/churn_v1.csv'))"
step "corrupt batch is rejected" "${PYTHON}" -c "
import sys; sys.path.insert(0,'src')
import pandas as pd, schema
try:
    schema.validate(pd.read_csv('data/churn_corrupt.csv')); sys.exit(1)
except schema.ValidationFailed: sys.exit(0)"
echo ""

# --- session 1 --------------------------------------------------------------
echo "  SESSION 1  reproducibility"
step "step1 the problem" "${PYTHON}" sessions/session01_reproducibility/step1_the_problem.py
step "step2 tracked training" "${PYTHON}" sessions/session01_reproducibility/step2_tracked_training.py
step "step3 reproduce (succeeds)" "${PYTHON}" sessions/session01_reproducibility/step3_reproduce.py --run-name monday
step "step3 diagnose divergence" "${PYTHON}" sessions/session01_reproducibility/step3_reproduce.py --run-name monday --data data/churn_v2.csv
echo ""

# --- session 2 --------------------------------------------------------------
echo "  SESSION 2  pipeline + registry"
step "pipeline happy path (registers v1)" "${PYTHON}" sessions/session02_pipeline/run_pipeline.py
step_expect_fail "validation gate halts on corrupt batch" "${PYTHON}" sessions/session02_pipeline/run_pipeline.py --data data/churn_corrupt.csv
step "challenger model trains + registers" "${PYTHON}" sessions/session02_pipeline/run_pipeline.py --model forest
echo ""

# --- session 3 --------------------------------------------------------------
echo "  SESSION 3  orchestration"
step "flow completes" "${PYTHON}" sessions/session03_orchestration/retraining_flow.py
step "transient fault recovered by retry" "${PYTHON}" sessions/session03_orchestration/retraining_flow.py --fail-at train --fail-times 2
echo ""

# --- session 4 --------------------------------------------------------------
echo "  SESSION 4  deployment"
step "export champion (logreg)" "${PYTHON}" sessions/session04_deployment/export_model.py --kind logreg --out champion
step "export challenger (forest)" "${PYTHON}" sessions/session04_deployment/export_model.py --kind forest --out challenger
step "start both services" bash sessions/session04_deployment/start_services.sh start
step "canary 90/10" "${PYTHON}" sessions/session04_deployment/canary.py --requests 60 --canary-weight 10
step "rollback to 100/0" "${PYTHON}" sessions/session04_deployment/canary.py --requests 40 --canary-weight 0
echo ""

# --- session 5 --------------------------------------------------------------
echo "  SESSION 5  monitoring"
step "drift injection against live service" "${PYTHON}" sessions/session05_monitoring/drift_injector.py
step "drift detection report" "${PYTHON}" sessions/session05_monitoring/drift_report.py
step "performance estimation (CBPE)" "${PYTHON}" sessions/session05_monitoring/performance_estimation.py
echo ""

# --- session 6 --------------------------------------------------------------
echo "  SESSION 6  observability"
step "silent-failure incident drill" "${PYTHON}" sessions/session06_observability/incident.py
echo ""

# --- aws (offline checks only) ----------------------------------------------
echo "  AWS  (syntax + argparse only; no credentials used)"
for f in aws/*.py; do
  step "$(basename "$f") parses" "${PYTHON}" "$f" --help
done
echo ""

# --- teardown ---------------------------------------------------------------
bash sessions/session04_deployment/start_services.sh stop >> "${LOG}" 2>&1 || true

echo "=========================================================================="
if [ "${FAIL}" -eq 0 ]; then
  echo "  ALL ${PASS} CHECKS PASSED -- you are ready to deliver"
  echo "=========================================================================="
  echo ""
  exit 0
else
  echo "  ${PASS} passed, ${FAIL} FAILED"
  echo "=========================================================================="
  echo ""
  for s in "${FAILED_STEPS[@]}"; do echo "    - ${s}"; done
  echo ""
  echo "  Details: ${LOG}"
  echo ""
  exit 1
fi
