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

# Must happen before numpy/sklearn are imported. Each GridSearchCV(n_jobs=-1) worker process
# also runs numpy's own BLAS calls, which by default multithread across every logical core -
# 22 joblib processes x 22 BLAS threads each oversubscribes a 22-thread machine ~22x over,
# and was observed to OOM-kill worker processes mid-fit (joblib silently respawns and retries,
# which is what turned a ~70min run into a 10+ hour one). Pin BLAS to 1 thread per worker so
# joblib's process-level parallelism is the only parallelism.
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import (GridSearchCV, KFold, StratifiedKFold,
                                     GroupKFold, StratifiedGroupKFold)
from joblib import parallel_backend
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                             mean_absolute_error, r2_score,
                             brier_score_loss, matthews_corrcoef)
from scipy.stats import spearmanr


# Searched (not fixed) inside the inner CV, same as every other hyperparameter below.
# Percentile rather than a raw feature count: dataset feature counts here range from 20
# (base) to 216 (biophases), and a fixed k would be either too small to matter for base or
# too large to do anything for biophases. Percentile keeps the same grid meaningful across
# every family - it is measured against whatever X.shape[1] actually is for that dataset.
# Only 2 levels, not a finer sweep: inner folds here have ~50 samples, and comparing more
# candidates against that little data doesn't resolve a real optimum, it mostly fits inner-CV
# noise (Cawley & Talbot 2010, JMLR - overfitting in model selection itself, independent of
# the leakage SelectPercentile-in-Pipeline already prevents).
FS_PERCENTILE_GRID = [50, 100]

# Same reasoning fixes RF's n_estimators rather than searching it (was the largest single
# combinatorial driver: 3 values x every other RF dimension): RF accuracy is well known to be
# flat above a few hundred trees, so searching it mostly bought noise, not a better model.
RF_N_ESTIMATORS = 500

# MLP's default (early_stopping=False) trains for the full max_iter regardless of whether it's
# still improving - combinations that don't converge quickly no longer just cost a bit of extra
# time each, they multiply into hours. early_stopping holds out part of the training fold to
# detect a plateau and stop early, which is also a legitimate regularizer, not just a speed hack.

# GridSearchCV's own parallelism, not to be confused with the BLAS-thread fix above (that
# controls threads *within* one worker; this controls how many worker *processes* run at once).
# This machine has 22 logical cores but as little as 3GB free RAM even at idle - 22 concurrent
# worker processes was observed to occasionally exceed that and get OOM-killed mid-fit (one
# nested-CV run took 3.8 hours instead of ~15min, and a later one crashed outright). Capping
# at 6 trades some wall-clock time for actually finishing reliably.
N_JOBS = 6


def build_models(task: str, seed: int, repeat_i: int = 0):
    """repeat_i only varies DummyClassifier's random_state (see call site in nested_cv): a
    fresh DummyClassifier(random_state=seed) built every repeat/fold, as this function is,
    gives byte-identical "random" predictions whenever the same test-fold shape recurs -
    which is constant with only 8-9 subjects - so the dummy baseline's repeats were not
    independent draws and its mean was biased well away from the 0.5 a stratified-guess
    baseline should have. Ridge/RF/MLP keep random_state=seed unvaried on purpose: it isolates
    how much performance varies because of the CV split from how much varies because of the
    model's own internal randomness (bootstrap sampling, weight init)."""
    if task == "regression":
        return {
            "dummy": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("m", DummyRegressor(strategy="mean"))]), {}),
            "ridge": (Pipeline([("imp", SimpleImputer(strategy="median")),
                                ("sc", StandardScaler()),
                                ("fs", SelectPercentile(f_regression)),
                                ("m", Ridge(random_state=seed))]),
                      {"fs__percentile": FS_PERCENTILE_GRID,
                       "m__alpha": [0.01, 0.1, 1, 10, 100, 300]}),
            "rf": (Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("fs", SelectPercentile(f_regression)),
                             ("m", RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, random_state=seed))]),
                   {"fs__percentile": FS_PERCENTILE_GRID,
                    "m__max_depth": [None, 3, 5, 10],
                    "m__min_samples_leaf": [1, 2, 4]}),
            "mlp": (Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler()),
                              ("fs", SelectPercentile(f_regression)),
                              ("m", MLPRegressor(max_iter=3000, early_stopping=True, random_state=seed))]),
                    {"fs__percentile": FS_PERCENTILE_GRID,
                     "m__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
                     "m__alpha": [0.001, 0.01, 0.1, 1.0]}),
        }
    return {
        "dummy": (Pipeline([("imp", SimpleImputer(strategy="median")),
                            ("m", DummyClassifier(strategy="stratified", random_state=seed + repeat_i))]), {}),
        "logreg": (Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("sc", StandardScaler()),
                             ("fs", SelectPercentile(f_classif)),
                             ("m", LogisticRegression(max_iter=2000, random_state=seed))]),
                   {"fs__percentile": FS_PERCENTILE_GRID,
                    "m__C": [0.01, 0.1, 1, 10]}),
        "rf": (Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("fs", SelectPercentile(f_classif)),
                         ("m", RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=seed))]),
               {"fs__percentile": FS_PERCENTILE_GRID,
                "m__max_depth": [None, 5, 10],
                "m__min_samples_leaf": [1, 2, 4]}),
        "mlp": (Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler()),
                          ("fs", SelectPercentile(f_classif)),
                          ("m", MLPClassifier(max_iter=3000, early_stopping=True, random_state=seed))]),
                {"fs__percentile": FS_PERCENTILE_GRID,
                 "m__hidden_layer_sizes": [(16,), (32,), (32, 16), (64, 32)],
                 "m__alpha": [0.001, 0.01, 0.1, 1.0]}),
    }


def _outer_splits(X, y, groups, task, outer, repeats, seed):
    """Yields (repeat_i, fold_i, train_idx, test_idx). When `groups` is given, uses
    Group(Stratified)KFold so no subject ever appears in both train and test. GroupKFold
    (regression) has no shuffle/random_state, so repeats are obtained by permuting row
    order per repeat (which fold a group lands in depends on group order/size)."""
    n = len(X)
    for repeat_i in range(repeats):
        rng = np.random.default_rng(seed + repeat_i)
        if groups is not None:
            if task == "classification":
                splitter = StratifiedGroupKFold(n_splits=outer, shuffle=True, random_state=seed + repeat_i)
                split_iter = splitter.split(X, y, groups=groups)
            else:
                perm = rng.permutation(n)
                splitter = GroupKFold(n_splits=outer)
                split_iter = ((perm[tr], perm[te]) for tr, te in
                              splitter.split(X.iloc[perm], y.iloc[perm], groups=groups.iloc[perm]))
        else:
            splitter = (StratifiedKFold(n_splits=outer, shuffle=True, random_state=seed + repeat_i)
                        if task == "classification"
                        else KFold(n_splits=outer, shuffle=True, random_state=seed + repeat_i))
            split_iter = splitter.split(X, y)
        for fold_i, (tr, te) in enumerate(split_iter):
            yield repeat_i, fold_i, tr, te


def nested_cv(X, y, task="regression", seed=42, outer=5, inner=3, repeats=10, groups=None, verbose=True):
    """Repeated nested CV: with ~75 samples a single outer split is noisy (see fold-to-fold
    swings of +/-0.2 in Spearman), so the outer split is repeated `repeats` times with different
    shuffles and all runs are pooled before averaging. Slower, but the mean/std it reports are
    actually trustworthy instead of an artifact of one lucky/unlucky split.

    If `groups` (e.g. subject id) is given, both outer and inner splits are group-aware
    (Group/StratifiedGroupKFold): no subject ever appears in both train and test, in either
    loop - the anti-leakage rule this kind of project lives or dies by.

    Returns {model_name: [per-fold metric dict, ...]} - the raw numbers, not a printed
    summary. Callers (main() here, or run_experiments.py) are responsible for saving or
    printing them; this function does not decide where results end up."""
    scoring = "neg_mean_absolute_error" if task == "regression" else "balanced_accuracy"
    results = {name: [] for name in build_models(task, seed)}

    # One persistent worker pool for the whole run (up to ~300 GridSearchCV.fit() calls: outer
    # x repeats x models), not one spun up and torn down per call - repeated pool creation is
    # itself a plausible source of the memory growth that showed up as "worker stopped...
    # memory leak" warnings and multi-hour blowups on later datasets in a sweep.
    with parallel_backend("loky", n_jobs=N_JOBS):
        for repeat_i, fold_i, tr, te in _outer_splits(X, y, groups, task, outer, repeats, seed):
            Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
            groups_tr = groups.iloc[tr] if groups is not None else None
            if groups is not None:
                inner_cv = list(_outer_splits(Xtr, ytr, groups_tr, task, inner, 1, seed + repeat_i))
                inner_cv = [(a, b) for _, _, a, b in inner_cv]
            else:
                inner_cv = (KFold(inner, shuffle=True, random_state=seed + repeat_i) if task == "regression"
                            else StratifiedKFold(inner, shuffle=True, random_state=seed + repeat_i))
            for name, (pipe, grid) in build_models(task, seed, repeat_i).items():
                gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner_cv)
                gs.fit(Xtr, ytr)
                pred = gs.predict(Xte)
                if task == "regression":
                    spearman_rho, _ = spearmanr(yte, pred)
                    m = {"mae": mean_absolute_error(yte, pred), "r2": r2_score(yte, pred),
                         "spearman": spearman_rho}
                else:
                    proba = gs.predict_proba(Xte)[:, 1]
                    try:
                        auc = roc_auc_score(yte, proba)
                    except ValueError:
                        # only a single class in yte (possible with tiny/imbalanced group-held-out
                        # folds) makes AUC undefined - anything else should surface, not hide as NaN
                        auc = np.nan
                    m = {"bal_acc": balanced_accuracy_score(yte, pred), "auc": auc,
                         # calibration (are predicted probabilities trustworthy, not just ranking)
                         "brier": brier_score_loss(yte, proba),
                         # single-number metric robust to the mild class imbalance some folds have
                         "mcc": matthews_corrcoef(yte, pred)}
                m["repeat"], m["fold"] = repeat_i + 1, fold_i + 1
                results[name].append(m)
                if verbose:
                    print(f"[repeat {repeat_i + 1} fold {fold_i + 1}] {name:8s} " +
                          "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k not in ("repeat", "fold")))

    if verbose:
        print(f"\nSummary (mean +/- std over {outer} x {repeats} = {outer * repeats} outer splits)")
        for name, folds in results.items():
            keys = [k for k in folds[0] if k not in ("repeat", "fold")]
            summ = "  ".join(
                f"{k}={np.nanmean([f[k] for f in folds]):.3f}+/-{np.nanstd([f[k] for f in folds]):.3f}"
                for k in keys)
            print(f"{name:8s} {summ}")
    return results


def results_to_frame(results: dict, **tags) -> pd.DataFrame:
    """Flattens nested_cv's {model: [fold_dict, ...]} into one row per (model, repeat, fold),
    with extra columns (e.g. dataset=..., task=...) attached - the shape a CSV/DataFrame needs."""
    rows = []
    for model_name, folds in results.items():
        for fold in folds:
            rows.append({**tags, "model": model_name, **fold})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=10,
                    help="how many times to repeat the outer CV split (default 10, slower but stabler)")
    p.add_argument("--groups", default=None,
                    help="column name (e.g. subject id) for group-aware CV: no group ever "
                         "appears in both train and test, in either the outer or inner loop")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    y = df[args.target]
    groups = df[args.groups] if args.groups else None
    drop_cols = [args.target] + ([args.groups] if args.groups else [])
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    group_msg = f", grouped by '{args.groups}' ({groups.nunique()} groups)" if groups is not None else ""
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features. Task: {args.task}{group_msg}")
    nested_cv(X, y, task=args.task, seed=args.seed, repeats=args.repeats, groups=groups)


if __name__ == "__main__":
    main()
