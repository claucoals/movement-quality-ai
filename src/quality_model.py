"""
Nested cross-validation for movement-quality models.

Two modes:
  --task regression      -> predict an expert quality score (e.g. KIMORE 0-100)
  --task classification  -> predict correct vs incorrect execution (e.g. UI-PRMD)

Outer loop  = unbiased performance estimate.  Inner loop = hyper-parameter search.
Reuses the RadiomicART nested-CV logic on kinematic features.

Usage:
    python quality_model.py --features feats.csv --target score --task regression
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             mean_absolute_error, r2_score)
from scipy.stats import spearmanr


def build_models(task: str, seed: int):
    if task == "regression":
        return {
            "dummy": (Pipeline([("m", DummyRegressor(strategy="mean"))]), {}),
            "ridge": (Pipeline([("sc", StandardScaler()),
                                ("m", Ridge(random_state=seed))]),
                      {"m__alpha": [0.1, 1, 10, 100]}),
            "rf": (Pipeline([("sc", StandardScaler()),
                             ("m", RandomForestRegressor(random_state=seed))]),
                   {"m__n_estimators": [300, 600], "m__max_depth": [None, 5, 10]}),
        }
    return {
        "logreg": (Pipeline([("sc", StandardScaler()),
                             ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
                   {"m__C": [0.01, 0.1, 1, 10]}),
        "rf": (Pipeline([("sc", StandardScaler()),
                         ("m", RandomForestClassifier(random_state=seed))]),
               {"m__n_estimators": [300, 600], "m__max_depth": [None, 5, 10]}),
    }


def nested_cv(X, y, task="regression", seed=42, outer=5, inner=3):
    scoring = "neg_mean_absolute_error" if task == "regression" else "balanced_accuracy"
    splitter = (KFold(outer, shuffle=True, random_state=seed) if task == "regression"
                else StratifiedKFold(outer, shuffle=True, random_state=seed))
    results = {name: [] for name in build_models(task, seed)}

    for fold, (tr, te) in enumerate(splitter.split(X, y), 1):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        inner_cv = (KFold(inner, shuffle=True, random_state=seed) if task == "regression"
                    else StratifiedKFold(inner, shuffle=True, random_state=seed))
        for name, (pipe, grid) in build_models(task, seed).items():
            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner_cv, n_jobs=-1)
            gs.fit(Xtr, ytr)
            pred = gs.predict(Xte)
            if task == "regression":
                m = {"mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred),
                     "spearman": spearmanr(yte, pred).correlation}
            else:
                try:
                    auc = roc_auc_score(yte, gs.predict_proba(Xte)[:, 1])
                except Exception:
                    auc = np.nan
                m = {"bal_acc": balanced_accuracy_score(yte, pred), "auc": auc}
            m["fold"] = fold
            results[name].append(m)
            print(f"[fold {fold}] {name:8s} " +
                  "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "fold"))

    print("\n=== Summary (mean +/- std over outer folds) ===")
    for name, folds in results.items():
        keys = [k for k in folds[0] if k != "fold"]
        summ = "  ".join(
            f"{k}={np.nanmean([f[k] for f in folds]):.3f}+/-{np.nanstd([f[k] for f in folds]):.3f}"
            for k in keys)
        print(f"{name:8s} {summ}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = pd.read_csv(args.features)
    y = df[args.target]
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features. Task: {args.task}")
    nested_cv(X, y, task=args.task, seed=args.seed)


if __name__ == "__main__":
    main()
