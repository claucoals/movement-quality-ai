"""
Gate 4 (deviation diagnostics): persists what univariate_check.py already computes for every
deviation__phase__joint feature (Mann-Whitney AUC + Cohen's d vs `correct`) as one reusable
CSV across all 6 REHAB24 exercises, instead of six separate stdout-only runs that nothing
keeps. No new statistics - same function univariate_check.py already exposes, just looped and
saved so notebooks can read it the same way they read results/experiments/experiments.csv and
results/shap/*.csv rather than re-deriving it by hand.

Usage:
    python src/run_univariate_deviation.py
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from univariate_check import run_classification

from paths import FEATURES, RESULTS

OUT_PATH = RESULTS / "univariate" / "rehab24_deviation.csv"


def main():
    frames = []
    for i in range(1, 7):
        exercise = f"ex{i}"
        path = FEATURES / "rehab24" / exercise / "deviation.csv"
        df = pd.read_csv(path)
        y = df["correct"]
        X = df.drop(columns=["correct", "subject"]).select_dtypes(include=[np.number])

        out = run_classification(X, y)
        out.insert(0, "exercise", exercise)
        n_sig = (out["p"] < 0.05).sum()
        top = out.iloc[0]
        print(f"{exercise}: {n_sig}/{len(out)} features p<0.05, strongest = {top['feature']} "
              f"(auc={top['auc']:.3f}, d={top['cohens_d']:.3f})")
        frames.append(out)

    combined = pd.concat(frames, ignore_index=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"\n{combined.shape[0]} feature rows (6 exercises) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
