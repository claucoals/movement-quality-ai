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
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     RepeatedKFold, RepeatedStratifiedKFold)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             mean_absolute_error, r2_score)
from scipy.stats import spearmanr


def build_models(task: str, seed: int):
    if task == "regression":
        return {
            "dummy": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", DummyRegressor(strategy="mean"))]), {}),
            "ridge": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler()),
                                ("m", Ridge(random_state=seed))]),
                      {"m__alpha": [0.01, 0.1, 1, 10, 100, 300]}),
            "rf": (Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("m", RandomForestRegressor(random_state=seed))]),
                   {"m__n_estimators": [300, 600, 900],
                    "m__max_depth": [None, 3, 5, 10],
                    "m__min_samples_leaf": [1, 2, 4]}),
            "mlp": (Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("m", MLPRegressor(max_iter=3000, random_state=seed))]),
                    {"m__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                     "m__alpha": [0.001, 0.01, 0.1, 1.0]}),
        }
    return {
        "dummy": (Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", DummyClassifier(strategy="stratified", random_state=seed))]), {}),
        "logreg": (Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
                   {"m__C": [0.01, 0.1, 1, 10]}),
        "rf": (Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("m", RandomForestClassifier(random_state=seed))]),
               {"m__n_estimators": [300, 600], "m__max_depth": [None, 5, 10]}),
        "mlp": (Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler()),
                          ("m", MLPClassifier(max_iter=3000, random_state=seed))]),
                {"m__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                 "m__alpha": [0.001, 0.01, 0.1, 1.0]}),
    }


def nested_cv(X, y, task="regression", seed=42, outer=5, inner=3, repeats=10):
    """Repeated nested CV: with ~75 samples a single outer split is noisy (see fold-to-fold
    swings of +/-0.2 in Spearman), so the outer split is repeated `repeats` times with different
    shuffles and all runs are pooled before averaging. Slower, but the mean/std it reports are
    actually trustworthy instead of an artifact of one lucky/unlucky split."""
    scoring = "neg_mean_absolute_error" if task == "regression" else "balanced_accuracy"
    outer_splitter = (RepeatedKFold(n_splits=outer, n_repeats=repeats, random_state=seed)
                       if task == "regression"
                       else RepeatedStratifiedKFold(n_splits=outer, n_repeats=repeats, random_state=seed))
    results = {name: [] for name in build_models(task, seed)}

    for split_i, (tr, te) in enumerate(outer_splitter.split(X, y), 1):
        repeat_i, fold_i = divmod(split_i - 1, outer)
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        inner_cv = (KFold(inner, shuffle=True, random_state=seed + repeat_i) if task == "regression"
                    else StratifiedKFold(inner, shuffle=True, random_state=seed + repeat_i))
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
            m["repeat"], m["fold"] = repeat_i + 1, fold_i + 1
            results[name].append(m)
            print(f"[repeat {repeat_i + 1} fold {fold_i + 1}] {name:8s} " +
                  "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k not in ("repeat", "fold")))

    print(f"\nSummary (mean +/- std over {outer} x {repeats} = {outer * repeats} outer splits)")
    for name, folds in results.items():
        keys = [k for k in folds[0] if k not in ("repeat", "fold")]
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
    p.add_argument("--repeats", type=int, default=10,
                    help="how many times to repeat the outer CV split (default 10, slower but stabler)")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    y = df[args.target]
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features. Task: {args.task}")
    nested_cv(X, y, task=args.task, seed=args.seed, repeats=args.repeats)


if __name__ == "__main__":
    main()
