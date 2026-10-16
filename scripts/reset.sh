#!/usr/bin/env bash
#
# Reset the demo to a clean state.
#
# RUN THIS BEFORE EVERY SESSION. Demos that have been run before carry state --
# an MLflow registry with nine versions, services already running, drift
# reports from last time. A clean start means the numbers on screen match the
# numbers in the playbook.
#
#   bash scripts/reset.sh          # clear generated state, keep data
#   bash scripts/reset.sh --all    # also regenerate the datasets
#
# What it clears:
#   - running champion/challenger services
#   - the MLflow store (experiments, runs, registry)
#   - Prefect's local database
#   - exported model artifacts
#   - monitoring reports and scored batches
#   - incident reports and service logs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
if [ -x "${REPO_ROOT}/../.venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/../.venv/bin/python"
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PYTHON="${REPO_ROOT}/.venv/bin/python"
fi

echo ""
echo "=========================================================================="
echo "  RESETTING DEMO STATE"
echo "=========================================================================="
echo ""

# --- 1. stop anything running ---------------------------------------------
echo "  stopping services..."
bash sessions/session04_deployment/start_services.sh stop > /dev/null 2>&1 || true
pkill -f "uvicorn app:app" 2>/dev/null || true
pkill -f "sagemaker_shim:app" 2>/dev/null || true
pkill -f "prefect server" 2>/dev/null || true
sleep 1

# --- 2. clear generated artifacts -----------------------------------------
echo "  clearing artifacts/..."
rm -rf artifacts/mlflow.db artifacts/mlartifacts artifacts/mlruns
rm -rf artifacts/prefect
rm -rf artifacts/serving
rm -rf artifacts/monitoring
rm -rf artifacts/incident
rm -rf artifacts/session01
rm -rf artifacts/logs artifacts/pids
mkdir -p artifacts

# --- 3. optionally regenerate data ----------------------------------------
if [ "${1:-}" = "--all" ]; then
  echo "  regenerating datasets..."
  "${PYTHON}" src/make_data.py | sed 's/^/    /'
fi

echo ""
echo "=========================================================================="
echo "  CLEAN"
echo "=========================================================================="
echo ""
echo "  Next:"
echo "    bash scripts/validate_all.sh      # full rehearsal, ~3 minutes"
echo "  or go straight to Session 1:"
echo "    python sessions/session01_reproducibility/step1_the_problem.py"
echo ""
