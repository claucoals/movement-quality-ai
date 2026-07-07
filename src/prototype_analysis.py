"""
The version of "prototype learning" REHAB24's data actually supports. Checked
data/raw/rehab24/annotations.csv directly: it has `correctness` (binary) and
`exercise_subtype` (which limb), no error-type label - so named archetypes
("Knee-Valgus Squat", "Trunk-Lean Squat") aren't buildable from this dataset alone. Those
only become possible once data with deliberate, labeled error types is collected.

What REHAB24 does support, and what this script does:

1. `composite_score` - the prototype itself isn't new: build_reference_deviation.py already
   builds it (the leave-subject-out average of correct reps' phase trajectories) and reports
   per-(phase, joint) distance to it. This turns that into the simplest possible classifier -
   a single "how far is this rep from the correct prototype, overall" number - and asks
   whether distance-to-prototype alone separates correct from incorrect reps, i.e. whether the
   most interpretable possible model is even in the right ballpark of the trained multivariate
   models in experiments.csv. Composite = mean of each deviation__phase__joint feature's
   within-exercise percentile rank, not raw values (the 36 features aren't comparable units -
   most are degrees, knee_valgus is a unitless ratio - percentile puts them on the same 0-1
   scale first, same fusion currency as build_attribution_map.py).

2. `cluster_incorrect` - exploratory only, flagged as such in its own output: unsupervised
   k-means over just the `incorrect` reps' deviation features, to see whether distinct error
   clusters emerge on their own without a label forcing them (loosely in the spirit of
   hard-negative framings in recent rehab-AQA contrastive work). With only ~20-60 incorrect
   reps per exercise this is descriptive, not a validated result - report it as "candidate
   clusters worth looking at with real error labels," not as discovered archetypes.

Usage:
    python src/prototype_analysis.py --exercise Ex6
    python src/prototype_analysis.py --all
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from univariate_check import cohens_d

from paths import FEATURES, RESULTS

SCORE_OUT = RESULTS / "prototype" / "rehab24_prototype_score.csv"
CLUSTER_OUT = RESULTS / "prototype" / "rehab24_incorrect_clusters.csv"
DEVIATION_DIAGNOSTICS = RESULTS / "univariate" / "rehab24_deviation.csv"


def load_deviation(exercise: str) -> pd.DataFrame:
    return pd.read_csv(FEATURES / "rehab24" / exercise / "deviation.csv")


def _score_and_test(score: pd.Series, correct: pd.Series) -> tuple[float, float, float]:
    g1 = score[correct == 1].values
    g0 = score[correct == 0].values
    u_stat, pval = mannwhitneyu(g1, g0)
    auc = u_stat / (len(g1) * len(g0))
    return auc, cohens_d(g1, g0), pval


def composite_score(exercise: str) -> dict:
    """Blanket average of every deviation feature's percentile - the naive version of
    'distance to the correct prototype'. Averaging in ~30 near-zero-signal features alongside
    the handful run_univariate_deviation.py found significant dilutes the composite toward
    noise, same dilution mechanism as the biophases SHAP ranking in build_attribution_map.py -
    measured directly below via `refined_auc`, not assumed."""
    df = load_deviation(exercise)
    feat_cols = [c for c in df.columns if c.startswith("deviation__")]
    score = df[feat_cols].rank(pct=True).mean(axis=1)
    auc, d, pval = _score_and_test(score, df["correct"])

    diagnostics = pd.read_csv(DEVIATION_DIAGNOSTICS)
    sig_feats = diagnostics.loc[(diagnostics["exercise"] == exercise) & (diagnostics["p"] < 0.05),
                                 "feature"].tolist()
    if sig_feats:
        refined = df[sig_feats].rank(pct=True).mean(axis=1)
        refined_auc, refined_d, refined_p = _score_and_test(refined, df["correct"])
    else:
        refined_auc = refined_d = refined_p = np.nan

    return {"exercise": exercise, "n": len(df),
            "n_correct": int((df["correct"] == 1).sum()), "n_incorrect": int((df["correct"] == 0).sum()),
            "naive_auc": auc, "naive_d": d, "naive_p": pval,
            "n_sig_features": len(sig_feats), "refined_auc": refined_auc,
            "refined_d": refined_d, "refined_p": refined_p}


def cluster_incorrect(exercise: str, k: int = 2) -> list[dict]:
    df = load_deviation(exercise)
    incorrect = df[df["correct"] == 0]
    feat_cols = [c for c in df.columns if c.startswith("deviation__")]
    if len(incorrect) < 2 * k:
        return []  # too few incorrect reps for k clusters to mean anything

    Xz = StandardScaler().fit_transform(incorrect[feat_cols])
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xz)

    rows = []
    for c in range(k):
        mask = labels == c
        # which deviation feature most separates this cluster from the rest of the
        # incorrect reps - the closest thing to a "candidate fault name" without a label
        cluster_mean = Xz[mask].mean(axis=0)
        other_mean = Xz[~mask].mean(axis=0)
        top_feat_idx = int(np.argmax(np.abs(cluster_mean - other_mean)))
        rows.append({"exercise": exercise, "cluster": c, "n": int(mask.sum()),
                     "candidate_feature": feat_cols[top_feat_idx],
                     "z_gap": float(cluster_mean[top_feat_idx] - other_mean[top_feat_idx])})
    return rows


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exercise", help="e.g. Ex6")
    g.add_argument("--all", action="store_true")
    p.add_argument("--k", type=int, default=2, help="clusters to try within incorrect reps")
    args = p.parse_args()
    exercises = [f"ex{i}" for i in range(1, 7)] if args.all else [args.exercise.lower()]

    scores = pd.DataFrame([composite_score(ex) for ex in exercises])
    print("Prototype-distance composite score (percentile-averaged, single scalar per rep) - "
          "naive (all 36 features) vs refined (significant features only):")
    fmt = {c: "{:.3f}".format for c in ("naive_auc", "naive_d", "refined_auc", "refined_d")}
    fmt.update({c: "{:.4f}".format for c in ("naive_p", "refined_p")})
    print(scores.to_string(index=False, formatters=fmt))
    SCORE_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(SCORE_OUT, index=False)
    print(f"-> {SCORE_OUT}")

    cluster_rows = [r for ex in exercises for r in cluster_incorrect(ex, k=args.k)]
    if cluster_rows:
        clusters = pd.DataFrame(cluster_rows)
        print(f"\nExploratory only (small N) - candidate clusters within `incorrect` reps, "
              f"k={args.k}:")
        print(clusters.to_string(index=False, formatters={"z_gap": "{:.2f}".format}))
        clusters.to_csv(CLUSTER_OUT, index=False)
        print(f"-> {CLUSTER_OUT}")


if __name__ == "__main__":
    main()
