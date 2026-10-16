set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0

ok ()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad ()  { echo "  FAIL  $1"; FAIL=$((FAIL+1)); }

cleanup () {
  docker rm -f mlops-demo-plain mlops-demo-sm >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo ""
echo "=========================================================================="
echo "  DOCKER VALIDATION"
echo "=========================================================================="
echo ""

# --- daemon ----------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
  echo "  Docker daemon not reachable."
  echo ""
  echo "  Start Docker Desktop (or Podman/Rancher Desktop) and re-run."
  echo "  If you use Podman:  alias docker=podman"
  echo ""
  exit 1
fi
ok "docker daemon reachable"

# --- model artifacts -------------------------------------------------------
if [ ! -f artifacts/serving/champion/model.joblib ]; then
  echo ""
  echo "  Missing artifacts/serving/champion/model.joblib"
  echo "  Export it first:"
  echo "    python sessions/session02_pipeline/run_pipeline.py"
  echo "    python sessions/session04_deployment/export_model.py --kind logreg --out champion"
  echo ""
  exit 1
fi
ok "champion model artifact present"

# --- build: plain image ----------------------------------------------------
echo ""
echo "  building plain serving image (60-120s)..."
if docker build -q -f sessions/session04_deployment/serving/Dockerfile \
     --build-arg MODEL_DIR=artifacts/serving/champion \
     -t churn-scorer:validate . >/dev/null 2>&1; then
  ok "plain image builds"
else
  bad "plain image builds"
fi

# --- run: plain image ------------------------------------------------------
docker rm -f mlops-demo-plain >/dev/null 2>&1 || true
if docker run -d --name mlops-demo-plain -p 18000:8000 \
     -e MODEL_VERSION=validate churn-scorer:validate >/dev/null 2>&1; then
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:18000/health >/dev/null 2>&1 && break
    sleep 1
  done
  if curl -sf http://127.0.0.1:18000/health >/dev/null 2>&1; then
    ok "plain image serves /health"
  else
    bad "plain image serves /health"
    docker logs mlops-demo-plain 2>&1 | tail -15
  fi

  if curl -sf -X POST http://127.0.0.1:18000/predict \
       -H 'Content-Type: application/json' \
       -d '{"customers":[{"tenure_months":3,"monthly_charges":95.5,"total_charges":280.0,"support_tickets":5,"avg_latency_ms":120.0,"contract_type":"month-to-month","payment_method":"electronic-check","has_fiber":1}]}' \
       >/dev/null 2>&1; then
    ok "plain image serves /predict"
  else
    bad "plain image serves /predict"
  fi
  docker rm -f mlops-demo-plain >/dev/null 2>&1
else
  bad "plain image runs"
fi

# --- build: sagemaker image ------------------------------------------------
echo ""
echo "  building SageMaker image..."
if docker build -q -f sessions/session04_deployment/serving/Dockerfile.sagemaker \
     --build-arg MODEL_DIR=artifacts/serving/champion \
     -t churn-scorer-sm:validate . >/dev/null 2>&1; then
  ok "sagemaker image builds"
else
  bad "sagemaker image builds"
fi

# --- run: sagemaker image --------------------------------------------------
docker rm -f mlops-demo-sm >/dev/null 2>&1 || true
if docker run -d --name mlops-demo-sm -p 18080:8080 \
     -e MODEL_VERSION=validate churn-scorer-sm:validate >/dev/null 2>&1; then
  for _ in $(seq 1 30); do
    curl -sf http://127.0.0.1:18080/ping >/dev/null 2>&1 && break
    sleep 1
  done
  if curl -sf http://127.0.0.1:18080/ping >/dev/null 2>&1; then
    ok "sagemaker image answers GET /ping (200)"
  else
    bad "sagemaker image answers GET /ping"
    docker logs mlops-demo-sm 2>&1 | tail -15
  fi

  if curl -sf -X POST http://127.0.0.1:18080/invocations \
       -H 'Content-Type: application/json' \
       -d '[{"tenure_months":3,"monthly_charges":95.5,"total_charges":280.0,"support_tickets":5,"avg_latency_ms":120.0,"contract_type":"month-to-month","payment_method":"electronic-check","has_fiber":1}]' \
       >/dev/null 2>&1; then
    ok "sagemaker image answers POST /invocations"
  else
    bad "sagemaker image answers POST /invocations"
  fi
  docker rm -f mlops-demo-sm >/dev/null 2>&1
else
  bad "sagemaker image runs"
fi

echo ""
echo "=========================================================================="
if [ "${FAIL}" -eq 0 ]; then
  echo "  ALL ${PASS} DOCKER CHECKS PASSED"
  echo "=========================================================================="
  echo ""
  exit 0
else
  echo "  ${PASS} passed, ${FAIL} FAILED"
  echo "=========================================================================="
  echo ""
  exit 1
fi
