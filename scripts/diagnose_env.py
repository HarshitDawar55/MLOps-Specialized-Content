"""
RUN
  python scripts/diagnose_env.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def header(text: str) -> None:
    print()
    print("=" * 76)
    print(f"  {text}")
    print("=" * 76)


def main() -> int:
    header("ENVIRONMENT")

    import sklearn

    print(f"\n  python       : {sys.version.split()[0]}")
    print(f"  numpy        : {np.__version__}")
    print(f"  scikit-learn : {sklearn.__version__}")
    print(f"  pandas       : {pd.__version__}")

    blas_name = "unknown"
    try:
        import numpy.__config__ as cfg

        info = cfg.show(mode="dicts")
        blas = info.get("Build Dependencies", {}).get("blas", {})
        blas_name = f"{blas.get('name')} {blas.get('version')}"
    except Exception:
        pass
    print(f"  BLAS         : {blas_name}")

    # ---------------------------------------------------------------- 1
    header("TEST 1  Does the data itself contain anything pathological?")

    import features

    data_path = ROOT / "data" / "churn_v1.csv"
    if not data_path.exists():
        print("\n  data/churn_v1.csv missing. Run: python src/make_data.py\n")
        return 1

    df = pd.read_csv(data_path)
    X = features.build_matrix(df)

    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(X)

    has_nan = bool(np.isnan(Xs).any())
    has_inf = bool(np.isinf(Xs).any())
    absmax = float(np.abs(Xs).max())

    print(f"\n  NaN present        : {has_nan}")
    print(f"  Inf present        : {has_inf}")
    print(f"  max |value|        : {absmax:.3f}   (float64 overflows near 1.8e308)")

    if has_nan or has_inf or absmax > 1e100:
        print("\n  -> The DATA is the problem. Stop here and investigate the features.")
        return 1
    print("\n  -> Data is clean. Nothing here can overflow a float64.")

    # ---------------------------------------------------------------- 2
    header("TEST 2  Does training actually trip a floating-point exception?")

    print("\n  Training with np.seterr(all='raise') -- the strictest possible")
    print("  setting, where ANY floating-point flag becomes a hard exception.")

    import modeling

    old = np.seterr(all="raise")
    genuine_fp_error = False
    metrics = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, metrics = modeling.train_and_eval(df)
        print("\n  -> Completed with NO floating-point exception raised.")
    except FloatingPointError as exc:
        genuine_fp_error = True
        print(f"\n  -> FloatingPointError: {exc}")
    finally:
        np.seterr(**old)

    # ---------------------------------------------------------------- 3
    header("TEST 3  Do the results match the reference run?")

    REFERENCE = {"roc_auc": 0.7513, "recall": 0.7198, "f1": 0.5214}

    if metrics is None:
        print("\n  Training did not complete; cannot compare.")
        matches = False
    else:
        print()
        matches = True
        for key, expected in REFERENCE.items():
            actual = metrics[key]
            ok = abs(actual - expected) < 1e-4
            matches &= ok
            print(
                f"  {key:<10} expected {expected}   got {actual}   "
                f"{'MATCH' if ok else 'DIFFERS'}"
            )

    # ---------------------------------------------------------------- verdict
    header("VERDICT")

    if genuine_fp_error:
        print("""
  REAL numerical problem.

  A floating-point exception fired under strict settings. Do NOT suppress
  the warnings -- the results cannot be trusted. Investigate the feature
  matrix and the solver before going any further.
""")
        return 1

    if matches:
        print("""
  SPURIOUS warnings from your BLAS.

  Training completes cleanly under the strictest floating-point settings,
  and the metrics match the reference run exactly. The computation is fine.

  What is happening: some BLAS builds (commonly MKL, which ships with
  Anaconda) leave FPU exception flags set from masked or padding lanes in
  their vectorised kernels. numpy reads those flags back after the matmul
  and reports them, even though no value in your result was affected. That
  is why all three flags -- divide-by-zero, overflow and invalid -- fire
  together on the same line: genuine overflow would not set divide-by-zero.

  This is cosmetic. Suppressing it is legitimate.

  TO SILENCE IT for the demos:
    The session scripts already call demo_env.quiet_blas_fp_warnings().
    If you are still seeing warnings, you are running an older copy --
    re-pull the repo.

  TO SILENCE IT in your own code:
    import numpy as np
    np.seterr(over='ignore', divide='ignore', invalid='ignore')

  TO SEE THEM AGAIN (e.g. to show the room what a noisy env looks like):
    MLOPS_DEMO_SHOW_FP_WARNINGS=1 python sessions/...

  OPTIONAL, UNRELATED, AND WORTH KNOWING:
    The feature matrix is rank-deficient by 2 -- each one-hot group sums to
    1, which is collinear with the intercept (the classic dummy-variable
    trap), and charges_per_month_tenure correlates 0.997 with
    monthly_charges. The L2 penalty absorbs both, which is why the results
    are stable and identical with or without the fix (AUC moves 0.7513 ->
    0.7514). It is NOT the cause of these warnings. Left as-is deliberately
    so the numbers in the playbooks stay correct -- but it is a good thing
    to mention in Session 2 when you discuss feature engineering.
""")
        return 0

    print("""
  INCONCLUSIVE.

  No floating-point exception fired, but the metrics differ from the
  reference. That points at a library-version difference rather than a
  numerical fault. Compare your versions above against requirements.txt
  before relying on the numbers printed in the playbooks.
""")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
