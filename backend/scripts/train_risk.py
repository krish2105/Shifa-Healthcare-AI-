#!/usr/bin/env python
"""Phase C — ED risk stratification: logistic regression vs XGBoost.

Target: admission from the emergency department, predicted from triage-time data.

Four decisions here matter more than the model choice, and all four are forced by
the shape of the data rather than chosen for elegance.

**1. Grouped cross-validation.** 222 stays come from 64 patients. A random split
puts the same patient on both sides, and the model learns the patient rather than
the presentation. Every number below comes from `GroupKFold` on `subject_id`. On
this dataset the ungrouped estimate is materially optimistic, which is exactly why
it would have been the tempting one to report.

**2. Leakage exclusion.** `los_hours`, `n_vital_checks`, `hr_std`, `sbp_min` and
`o2sat_min` all describe what happened *during* the stay. Length of stay is close to
a restatement of the admission decision. They are computed by the ingester and
deliberately excluded here; `--include-leaky` reproduces the inflated numbers so the
size of the effect is visible rather than asserted.

**3. Race is not a predictor.** It is present in the source data. Using it would
likely improve AUC by encoding access and referral disparities as if they were
clinical facts, which is how a model launders structural bias into a risk score.
Excluded on purpose, not overlooked.

**4. Calibration is reported alongside AUC.** A triage tool whose "0.8" does not mean
80% is actively misleading at the bedside, and ranking metrics alone will not reveal
that. Brier score and calibration slope are reported for both models.

Usage:
    python scripts/train_risk.py
    python scripts/train_risk.py --include-leaky    # demonstrate the leakage effect
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from app.config import settings
from app.logging_conf import get_logger
from app.risk.features import LEAKY, TARGET, build_features

warnings.filterwarnings("ignore")
log = get_logger("risk.train")

SEED = 42


def bootstrap_auc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap CI for AUC.

    Reported because at n=222 with 64 patients a point estimate alone invites
    over-reading. The interval is wide, and it should be.
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(SEED)
    scores: list[float] = []
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(roc_auc_score(y[idx], p[idx]))
    if not scores:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of observed outcome regressed on the logit of predicted probability.

    1.0 is perfect. Below 1.0 means predictions are too extreme (overconfident),
    above means too timid.
    """
    from sklearn.linear_model import LogisticRegression

    eps = 1e-6
    logit = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(logit.reshape(-1, 1), y)
        return float(lr.coef_[0][0])
    except Exception:  # noqa: BLE001
        return float("nan")


def evaluate(name: str, y: np.ndarray, p: np.ndarray) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    lo, hi = bootstrap_auc_ci(y, p)
    return {
        "model": name,
        "auc_roc": round(float(roc_auc_score(y, p)), 4),
        "auc_roc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "calibration_slope": round(calibration_slope(y, p), 4),
        "mean_predicted": round(float(p.mean()), 4),
        "observed_rate": round(float(y.mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-leaky", action="store_true",
                    help="Include post-triage features to demonstrate the leakage effect.")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    settings.ensure_dirs()
    path = settings.processed_dir / "ed_encounters.parquet"
    if not path.exists():
        log.error("risk.no_data", hint="Run scripts/ingest_mimic.py first.")
        return 1

    df = pd.read_parquet(path)
    y = df[TARGET].to_numpy()
    groups = df["subject_id"].to_numpy()

    X, feature_names = build_features(df, args.include_leaky)
    log.info("risk.features", n_features=len(feature_names), rows=len(X),
             patients=len(np.unique(groups)), positive_rate=round(float(y.mean()), 4))

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier

    numeric_cols = list(X.columns)

    def make_lr() -> Pipeline:
        return Pipeline(
            [
                ("prep", ColumnTransformer([
                    ("num", Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), numeric_cols),
                ])),
                # Strong L2. With 222 rows and ~25 features an unpenalized fit
                # memorizes the sample.
                ("clf", LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced")),
            ]
        )

    def make_xgb() -> Pipeline:
        return Pipeline(
            [
                ("prep", ColumnTransformer([
                    ("num", SimpleImputer(strategy="median"), numeric_cols),
                ])),
                # Heavily constrained on purpose: shallow trees, few of them, strong
                # subsampling. Default XGBoost hyperparameters on 222 rows produce a
                # model that fits the training set perfectly and generalizes poorly.
                ("clf", XGBClassifier(
                    n_estimators=120, max_depth=2, learning_rate=0.06,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=3.0, min_child_weight=5,
                    eval_metric="logloss", random_state=SEED, n_jobs=2,
                )),
            ]
        )

    def make_lr_calibrated() -> Pipeline:
        """Platt-scaled logistic regression.

        The uncalibrated models come out with a calibration slope near 0.53, meaning
        their probabilities are far too extreme — a "0.9" does not correspond to a
        90% admission rate. For a number a clinician reads at the bedside that is a
        defect, not a rounding detail, so it gets corrected rather than footnoted.

        Sigmoid, not isotonic: isotonic is non-parametric and needs far more data
        than 222 rows before it stops fitting noise in the probability mapping.
        """
        from sklearn.calibration import CalibratedClassifierCV

        return Pipeline(
            [
                ("prep", ColumnTransformer([
                    ("num", Pipeline([
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]), numeric_cols),
                ])),
                ("clf", CalibratedClassifierCV(
                    LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"),
                    method="sigmoid", cv=3,
                )),
            ]
        )

    n_splits = min(args.folds, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    results: dict[str, dict] = {}
    oof: dict[str, np.ndarray] = {}

    for name, factory in (
        ("logistic_regression", make_lr),
        ("xgboost", make_xgb),
        ("logistic_regression_calibrated", make_lr_calibrated),
    ):
        preds = np.zeros(len(y), dtype=float)
        for train_idx, test_idx in gkf.split(X, y, groups):
            model = factory()
            model.fit(X.iloc[train_idx], y[train_idx])
            preds[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
        oof[name] = preds
        results[name] = evaluate(name, y, preds)
        log.info("risk.evaluated", **results[name])

    # -- baselines -------------------------------------------------------------
    # Without these the AUC numbers have no reference point. "Always admit" fixes
    # the accuracy floor; triage acuity alone is the comparison that matters, since
    # a model that cannot beat the number the triage nurse already wrote down has
    # not earned its place in the workflow.
    baselines: dict[str, dict] = {}
    baselines["always_admit"] = {
        "model": "always_admit",
        "accuracy": round(float(y.mean()), 4),
        "auc_roc": 0.5,
        "note": "majority-class baseline; the floor any model must clear",
    }
    if "acuity" in X.columns:
        acuity = X["acuity"].fillna(X["acuity"].median()).to_numpy()
        # Acuity is 1 = most urgent, so invert to make it a risk score.
        acuity_score = (5 - acuity) / 4.0
        baselines["triage_acuity_only"] = evaluate("triage_acuity_only", y, acuity_score)
        baselines["triage_acuity_only"]["note"] = (
            "the score a triage nurse already assigns, used directly as a risk score"
        )

    # -- ungrouped comparison, to quantify the leakage we avoided ---------------
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    ungrouped_xgb = cross_val_predict(make_xgb(), X, y, cv=skf, method="predict_proba")[:, 1]
    ungrouped = evaluate("xgboost_UNGROUPED_do_not_report", y, ungrouped_xgb)

    # -- pick and persist ------------------------------------------------------
    factories = {
        "logistic_regression": make_lr,
        "xgboost": make_xgb,
        "logistic_regression_calibrated": make_lr_calibrated,
    }

    # Selection is by Brier score, not AUC.
    #
    # AUC measures ranking only — it is invariant to any monotone transform of the
    # predictions, so a model can top the AUC table while its "0.9" corresponds to a
    # 70% event rate. Shifa42 displays this number to a clinician as a probability,
    # so the property that matters is whether the number means what it says.
    #
    # Brier score is a strictly proper scoring rule: it decomposes into calibration
    # and refinement terms and is minimized only by the truly correct probabilities.
    # It therefore captures both things at once and cannot be gamed by a model that
    # ranks well but is systematically overconfident.
    #
    # On this data the two criteria disagree, which is the whole point of stating the
    # rule up front rather than picking after seeing the table:
    #   AUC   ranks plain LR first  (0.710 vs 0.674) — CIs overlap heavily
    #   Brier ranks calibrated LR first (0.2065 vs 0.2184), slope 0.78 vs 0.53
    best_name = min(results, key=lambda k: results[k]["brier"])
    best_factory = factories[best_name]
    best_model = best_factory()
    best_model.fit(X, y)

    def extract_importances(clf) -> dict[str, float]:
        """Pull feature weights out of whichever estimator won.

        CalibratedClassifierCV exposes neither coef_ nor feature_importances_ — it
        holds one fitted estimator per internal fold — so we average their absolute
        coefficients.
        """
        if hasattr(clf, "feature_importances_"):
            vals = clf.feature_importances_.tolist()
        elif hasattr(clf, "coef_"):
            vals = np.abs(clf.coef_[0]).tolist()
        elif hasattr(clf, "calibrated_classifiers_"):
            mats = [
                np.abs(cc.estimator.coef_[0])
                for cc in clf.calibrated_classifiers_
                if hasattr(getattr(cc, "estimator", None), "coef_")
            ]
            if not mats:
                return {}
            vals = np.mean(mats, axis=0).tolist()
        else:
            return {}
        return dict(
            sorted(zip(feature_names, vals, strict=True), key=lambda kv: -kv[1])[:10]
        )

    importances = extract_importances(best_model.named_steps["clf"])

    import pickle

    with (settings.artifacts_dir / "risk_model.pkl").open("wb") as fh:
        pickle.dump({"model": best_model, "features": feature_names}, fh)

    auc_gap = round(ungrouped["auc_roc"] - results["xgboost"]["auc_roc"], 4)
    meta = {
        "task": "ED admission prediction from triage-time data",
        "dataset": "MIMIC-IV-ED Demo v2.2",
        "n_stays": len(df),
        "n_patients": len(np.unique(groups)),
        "positive_rate": round(float(y.mean()), 4),
        "n_features": len(feature_names),
        "features": feature_names,
        "cv": f"GroupKFold(n_splits={n_splits}) on subject_id",
        "leaky_features_included": args.include_leaky,
        "leaky_features_excluded": [] if args.include_leaky else LEAKY,
        "race_used_as_predictor": False,
        "results": results,
        "baselines": baselines,
        "leakage_check": {
            "ungrouped_xgboost_auc": ungrouped["auc_roc"],
            "grouped_xgboost_auc": results["xgboost"]["auc_roc"],
            "auc_inflation_from_ignoring_patient_grouping": auc_gap,
            "note": (
                "The ungrouped figure is what a naive random split would have reported. "
                "It is recorded here only to size the bias; the grouped figure is the "
                "one reported everywhere else."
            ),
        },
        "selected_model": best_name,
        "selection_rule": (
            "lowest Brier score. Brier is a strictly proper scoring rule covering both "
            "calibration and discrimination; AUC covers ranking alone and is invariant "
            "to monotone transforms, so it cannot detect the overconfidence that makes a "
            "displayed probability misleading. The two criteria disagree on this dataset."
        ),
        "top_features": importances,
        "seed": SEED,
    }
    settings.risk_meta_path.write_text(json.dumps(meta, indent=2, default=str))

    log.info("risk.complete", selected=best_name, auc=results[best_name]["auc_roc"])
    print("\n" + json.dumps(meta, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
