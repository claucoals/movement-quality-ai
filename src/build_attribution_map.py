"""
Gate 5 / Milestone 4: Movement Attribution Map (importance x deviation).

Fuses two things already computed by earlier gates into the phase x joint matrix the whole
project is aimed at (see next_phase_plan.md section 3):
  - Pipeline A, "what does the model rely on": results/shap/rehab24_biophases.csv, out-of-fold
    SHAP per (phase, feature) - see run_shap.py.
  - Pipeline B, "how far from correct technique": results/univariate/rehab24_deviation.csv,
    Mann-Whitney AUC + Cohen's d per (phase, joint) - see run_univariate_deviation.py.

Neither input is touched by the dummy-classifier bias fix in quality_model.py (that only
affected the dummy baseline's own reported numbers, never which of logreg/rf/mlp gets picked
as a fold's winner) - so this can run against the existing SHAP output without waiting for the
full resweep to finish.

Percentile rank, not raw magnitude, is the fusion currency on both sides: SHAP magnitude is
not comparable across feature-family sizes (splitting knee_valgus_min into 3 phase-features
alone drops its rank from 1/86 to 8/216 by dilution alone, see MODEL_CARD's Gate 3 note) -
percentile-within-exercise cancels that out. That is exactly why the vision doc specifies
percentile(SHAP) x percentile(deviation) rather than raw values - confirmed necessary here,
not just a nice default.

Both sides are aggregated from per-feature to per-joint by stripping the L/R split and the
summary-stat suffix (_min/_max/_rom/_mean/_std/_vel_mean_abs): the matrix this project's
feedback should read from is "which joint, which phase", not "which of 7 derived stats of
that joint's angle signal" - see JOINT_OF below.

Usage:
    python src/build_attribution_map.py --exercise Ex6
    python src/build_attribution_map.py --all
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from paths import RESULTS

SHAP_PATH = RESULTS / "shap" / "rehab24_biophases.csv"
DEVIATION_PATH = RESULTS / "univariate" / "rehab24_deviation.csv"
OUT_DIR = RESULTS / "attribution_map"

# Ordered longest-prefix-first: a feature stem (after stripping phase/left-right/sym_) is
# matched against these to decide which joint it belongs to on the human-readable matrix.
# "knee_valgus" must be checked before "knee", "shoulder_angle" before "shoulder".
JOINT_PATTERNS = [
    ("knee_valgus", "knee"),
    ("shoulder_angle", "shoulder"),
    ("shoulder", "shoulder"),
    ("elbow", "elbow"),
    ("knee", "knee"),
    ("hip", "hip"),
    ("ankle", "ankle"),
    ("trunk_flex", "trunk"),
]


def joint_of(stem: str) -> str | None:
    s = stem
    if s.startswith("sym_"):
        s = s[len("sym_"):]
    elif s.startswith(("l_", "r_")):
        s = s[2:]
    for prefix, joint in JOINT_PATTERNS:
        if s.startswith(prefix):
            return joint
    return None


def shap_importance(exercise: str) -> pd.DataFrame:
    """Mean|SHAP| per (phase, joint) for one exercise, from the out-of-fold biophases SHAP
    values every sample already has (run_shap.py --all --family biophases)."""
    df = pd.read_csv(SHAP_PATH)
    g = df[df["exercise"] == exercise]
    if g.empty:
        raise SystemExit(f"No biophases SHAP rows for {exercise} in {SHAP_PATH} - "
                          f"run: python src/run_shap.py --all --family biophases")
    shap_cols = [c for c in g.columns if c.startswith("shap__")]
    mean_abs = g[shap_cols].abs().mean()

    rows = []
    for col, val in mean_abs.items():
        # shap__{phase}__{rest}
        _, phase, rest = col.split("__", 2)
        joint = joint_of(rest)
        if joint is None:
            continue
        rows.append({"phase": phase, "joint": joint, "shap_mean_abs": val})
    out = pd.DataFrame(rows).groupby(["phase", "joint"], as_index=False)["shap_mean_abs"].mean()
    out["importance_percentile"] = out["shap_mean_abs"].rank(pct=True)
    return out


def deviation_signal(exercise: str) -> pd.DataFrame:
    """Mean|Cohen's d| per (phase, joint) for one exercise, from Gate 4's saved univariate
    deviation diagnostics (run_univariate_deviation.py)."""
    df = pd.read_csv(DEVIATION_PATH)
    g = df[df["exercise"] == exercise.lower()]
    if g.empty:
        raise SystemExit(f"No deviation univariate rows for {exercise} in {DEVIATION_PATH} - "
                          f"run: python src/run_univariate_deviation.py")

    rows = []
    for _, row in g.iterrows():
        # deviation__{phase}__{angle_name}
        _, phase, angle_name = row["feature"].split("__", 2)
        joint = joint_of(angle_name)
        if joint is None:
            continue
        rows.append({"phase": phase, "joint": joint, "abs_cohens_d": abs(row["cohens_d"])})
    out = pd.DataFrame(rows).groupby(["phase", "joint"], as_index=False)["abs_cohens_d"].mean()
    out["deviation_percentile"] = out["abs_cohens_d"].rank(pct=True)
    return out


def attribution_map(exercise: str) -> pd.DataFrame:
    imp = shap_importance(exercise)
    dev = deviation_signal(exercise)
    merged = imp.merge(dev, on=["phase", "joint"], how="inner")
    merged["fusion_score"] = merged["importance_percentile"] * merged["deviation_percentile"]
    merged.insert(0, "exercise", exercise)
    return merged.sort_values("fusion_score", ascending=False).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exercise", help="e.g. Ex6")
    g.add_argument("--all", action="store_true", help="run all 6 REHAB24 exercises")
    args = p.parse_args()

    exercises = [f"Ex{i}" for i in range(1, 7)] if args.all else [args.exercise]

    frames = []
    for ex in exercises:
        m = attribution_map(ex)
        top = m.iloc[0]
        print(f"[{ex}] largest quality loss: {top['joint']} / {top['phase']} "
              f"(fusion={top['fusion_score']:.3f}, "
              f"importance_pct={top['importance_percentile']:.2f}, "
              f"deviation_pct={top['deviation_percentile']:.2f})")
        frames.append(m)

    combined = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / ("rehab24_attribution_map.csv" if args.all
                           else f"rehab24_{args.exercise.lower()}_attribution_map.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n{combined.shape[0]} (phase, joint) cells -> {out_path}")


if __name__ == "__main__":
    main()
