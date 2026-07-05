"""
Concatenate per-exercise anatomical feature tables for leave-one-exercise-out CV.

Reads data/features/rehab24/ex{1..6}/anatomical.csv (built by
build_features_rehab24_anatomical.py), adds an `exercise` column, and writes
data/features/rehab24/pooled_anatomical.csv.

Usage:
    python src/build_features_rehab24_pooled.py
"""

from __future__ import annotations

import pandas as pd

from paths import rehab24_features, rehab24_pooled_anatomical


def main():
    frames = []
    for i in range(1, 7):
        ex = f"ex{i}"
        path = rehab24_features(ex, "anatomical")
        df = pd.read_csv(path)
        df["exercise"] = ex
        frames.append(df)
        print(f"{ex}: {df.shape[0]} righe da {path}")

    out = pd.concat(frames, ignore_index=True)
    dest = rehab24_pooled_anatomical()
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(f"\n{out.shape[0]} righe totali, {out['exercise'].nunique()} esercizi -> {dest}")


if __name__ == "__main__":
    main()
