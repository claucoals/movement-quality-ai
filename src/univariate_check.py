"""
Univariate check: does any single feature correlate with cTS on its own,
before blaming the model. No CV, no pipeline, just Spearman per feature.

Usage:
    python src/univariate_check.py --features data/features_ex5.csv --target cTS
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--target", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.features)
    y = df[args.target]
    X = df.drop(columns=[args.target]).select_dtypes(include=[np.number])
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    rows = []
    for col in X.columns:
        mask = X[col].notna()
        if mask.sum() < 3:
            continue
        rho, pval = spearmanr(X.loc[mask, col], y.loc[mask])
        rows.append({"feature": col, "n": int(mask.sum()), "rho": rho, "p": pval})

    out = pd.DataFrame(rows).sort_values("rho", key=lambda s: s.abs(), ascending=False)
    print(out.to_string(index=False, formatters={"rho": "{:.3f}".format, "p": "{:.3f}".format}))

    n_sig = (out["p"] < 0.05).sum()
    print(f"\nFeature con p < 0.05 (non corretto per test multipli): {n_sig} su {len(out)}")


if __name__ == "__main__":
    main()
