"""
Univariate check: does any single feature correlate with the target on its own, before
blaming the model. No CV, no pipeline - just one statistical test per feature.

--task regression (continuous target, e.g. KIMORE cTS): Spearman rho per feature.
--task classification (binary target, e.g. REHAB24 correct/incorrect): Mann-Whitney U per
feature, which gives both a p-value and an AUC in the same test (U / (n1*n0) is exactly the
AUC of using that one feature as a threshold classifier - no need for a separate ranking
metric), plus Cohen's d for the effect size.

Usage:
    python src/univariate_check.py --features data/features/kimore/ex5.csv --target cTS --task regression
    python src/univariate_check.py --features data/features/rehab24/ex1/deviation.csv --target correct --task classification
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(((n_a - 1) * a.std(ddof=1) ** 2 + (n_b - 1) * b.std(ddof=1) ** 2)
                          / (n_a + n_b - 2))
    return (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else np.nan


def run_regression(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    for col in X.columns:
        mask = X[col].notna()
        if mask.sum() < 3:
            continue
        rho, pval = spearmanr(X.loc[mask, col], y.loc[mask])
        rows.append({"feature": col, "n": int(mask.sum()), "rho": rho, "p": pval})
    return pd.DataFrame(rows).sort_values("rho", key=lambda s: s.abs(), ascending=False)


def run_classification(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    classes = sorted(y.dropna().unique())
    if len(classes) != 2:
        raise SystemExit(f"--task classification needs a binary target, got classes {classes}")
    rows = []
    for col in X.columns:
        mask = X[col].notna()
        g1 = X.loc[mask & (y == classes[1]), col].values
        g0 = X.loc[mask & (y == classes[0]), col].values
        if len(g1) < 3 or len(g0) < 3:
            continue
        u_stat, pval = mannwhitneyu(g1, g0)
        auc = u_stat / (len(g1) * len(g0))
        d = cohens_d(g1, g0)
        rows.append({"feature": col, "n": len(g1) + len(g0), "auc": auc, "cohens_d": d, "p": pval})
    return pd.DataFrame(rows).sort_values("auc", key=lambda s: (s - 0.5).abs(), ascending=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--task", choices=["regression", "classification"], default="regression")
    args = p.parse_args()

    df = pd.read_csv(args.features)
    y = df[args.target]
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    if args.task == "regression":
        out = run_regression(X, y)
        print(out.to_string(index=False, formatters={"rho": "{:.3f}".format, "p": "{:.3f}".format}))
    else:
        out = run_classification(X, y)
        print(out.to_string(index=False, formatters={"auc": "{:.3f}".format,
                                                       "cohens_d": "{:.3f}".format,
                                                       "p": "{:.3f}".format}))

    n_sig = (out["p"] < 0.05).sum()
    print(f"\nFeature con p < 0.05 (non corretto per test multipli): {n_sig} su {len(out)}")


if __name__ == "__main__":
    main()
