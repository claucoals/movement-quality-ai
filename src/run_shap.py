"""
Out-of-fold SHAP values for a REHAB24-6 feature family (anatomical or biophases).

For each exercise this picks whichever model type actually won that exercise+family in
results/experiments/experiments.csv (read, not hardcoded - if a rerun of the sweep changes the winner,
this script follows it automatically), then explains it with model-agnostic permutation SHAP
(shap.Explainer given the pipeline's predict_proba) - the same explainer code path for
logreg, rf and mlp, no per-model-type special casing.

Anti-leakage rule carried over from quality_model.py: every sample is explained only by a
model that never saw it during training. This uses the same grouped outer/inner split logic
(_outer_splits) as the main sweep, fits the winning model type per fold via GridSearchCV
(same grids as quality_model.build_models), and computes SHAP values on that fold's held-out
subjects only. Repeated across `repeats` outer splits and averaged per sample, so the result
isn't an artifact of one arbitrary fold assignment - explaining a model on its own training
data would overstate how meaningful the attributions are, exactly like scoring one would.

Usage:
    python src/run_shap.py --exercise Ex1 --family anatomical
    python src/run_shap.py --all --family biophases
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from shap.maskers import Independent as IndependentMasker
from sklearn.model_selection import GridSearchCV

from quality_model import build_models, _outer_splits, N_JOBS

from paths import EXPERIMENTS_CSV, SHAP_DIR, rehab24_features

ROOT = Path(__file__).resolve().parents[1]
OUTER = 5
INNER = 3
# Lower than the main sweep's repeats=20: each SHAP explanation is many more model
# evaluations than a single fit+predict (permutation SHAP re-evaluates the model per
# feature per sample), so full parity isn't practical. repeats=5 still means every sample
# gets explained by 5 independently-split, independently-trained models instead of just
# one, so the averaged attribution isn't a fold-assignment artifact.
REPEATS = 5
SEED = 42
# How many of the overall top features to check for rank stability across repeats (M5,
# next_phase_plan.md section 4). 10 rather than every feature: with only REPEATS=5 rankers,
# asking whether hundreds of near-zero-importance features keep a stable order is asking the
# metric to resolve noise, not signal - the top-10 is the part of the ranking anyone would
# actually read off a report.
TOP_K_STABILITY = 10


def kendalls_w(rank_matrix: np.ndarray) -> float:
    """Kendall's coefficient of concordance for `rank_matrix` (n_items, n_raters): 1.0 means
    every rater (here, every repeat's independently-trained model) ranked the items in
    exactly the same order, 0 means no more agreement than chance. Standard way to report
    SHAP-ranking stability across resamples (e.g. bootstrap-SHAP stability studies use the
    same statistic) rather than an ad hoc stability notion."""
    n, m = rank_matrix.shape
    R = rank_matrix.sum(axis=1)
    S = np.sum((R - R.mean()) ** 2)
    return float(12 * S / (m ** 2 * (n ** 3 - n)))


def pick_winning_model(exercise: str, family: str) -> str:
    res = pd.read_csv(EXPERIMENTS_CSV)
    dataset_name = f"rehab24_{exercise.lower()}_{family}"
    sub = res[(res["dataset"] == dataset_name) & (res["model"] != "dummy")]
    if sub.empty:
        raise SystemExit(f"No results for {dataset_name} in {EXPERIMENTS_CSV} - "
                          f"run src/run_experiments.py first.")
    means = sub.groupby("model")["auc"].mean()
    return str(means.idxmax())


def run_exercise(exercise: str, family: str) -> tuple[pd.DataFrame, dict]:
    path = rehab24_features(exercise, family)
    df = pd.read_csv(path)
    y = df["correct"]
    groups = df["subject"]
    X = df.drop(columns=["correct", "subject"]).select_dtypes(include=[np.number]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    feature_names = X.columns.tolist()
    n = len(X)

    model_name = pick_winning_model(exercise, family)
    print(f"[{exercise}] {n} samples, {groups.nunique()} subjects, winning model: {model_name}")

    pipe_template, grid = build_models("classification", SEED)[model_name]

    shap_sum = np.zeros((n, len(feature_names)))
    shap_count = np.zeros(n)
    # kept per-repeat (not just pooled into shap_sum) so stability across repeats can be
    # checked (M5) - every sample gets exactly one row filled per repeat, since each repeat's
    # OUTER folds partition the full sample set once.
    shap_by_repeat = np.zeros((REPEATS, n, len(feature_names)))
    n_folds_done = 0

    for repeat_i, fold_i, tr, te in _outer_splits(X, y, groups, "classification", OUTER, REPEATS, SEED):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr]
        groups_tr = groups.iloc[tr]

        inner_splits = list(_outer_splits(Xtr, ytr, groups_tr, "classification", INNER, 1, SEED + repeat_i))
        inner_cv = [(a, b) for _, _, a, b in inner_splits]

        gs = GridSearchCV(pipe_template, grid, scoring="balanced_accuracy", cv=inner_cv, n_jobs=N_JOBS)
        gs.fit(Xtr, ytr)
        best_pipe = gs.best_estimator_

        def predict_fn(arr, _pipe=best_pipe):
            return _pipe.predict_proba(pd.DataFrame(arr, columns=pd.Index(feature_names)))[:, 1]

        masker = IndependentMasker(Xtr.values, max_samples=Xtr.shape[0])
        explainer = shap.Explainer(predict_fn, masker, feature_names=feature_names)
        sv = explainer(Xte.values)
        assert isinstance(sv, shap.Explanation), f"expected a single Explanation, got {type(sv)}"

        shap_sum[te] += sv.values
        shap_count[te] += 1
        shap_by_repeat[repeat_i, te] = sv.values
        n_folds_done += 1
        print(f"  repeat {repeat_i + 1} fold {fold_i + 1}: best_params={gs.best_params_}, "
              f"explained {len(te)} held-out samples")

    assert (shap_count == REPEATS).all(), "every sample must be held out exactly once per repeat"
    mean_shap = shap_sum / shap_count[:, None]

    out = pd.DataFrame(mean_shap, columns=pd.Index(f"shap__{c}" for c in feature_names))
    out.insert(0, "exercise", exercise)
    out.insert(1, "subject", groups.values)
    out.insert(2, "correct", y.values)
    out.insert(3, "model", model_name)
    for c in feature_names:
        out[f"value__{c}"] = X[c].values
    print(f"  -> {n} samples explained ({n_folds_done} fold fits total)")

    overall_importance = np.abs(mean_shap).mean(axis=0)
    top_idx = np.argsort(overall_importance)[::-1][:TOP_K_STABILITY]
    # rank (1 = most important) each repeat's own feature ordering, restricted to the
    # features that are top-K overall - asks "do repeats agree on the order of the features
    # that matter", not "do repeats agree on hundreds of near-zero features".
    per_repeat_importance = np.abs(shap_by_repeat).mean(axis=1)  # (REPEATS, n_features)
    ranks = np.apply_along_axis(lambda v: len(v) - np.argsort(np.argsort(v)), 1, per_repeat_importance)
    rank_matrix = ranks[:, top_idx].T  # (TOP_K_STABILITY, REPEATS)
    w = kendalls_w(rank_matrix)
    stability = {"exercise": exercise, "family": family, "model": model_name,
                 "n_repeats": REPEATS, "top_k": len(top_idx), "kendalls_w": w,
                 "top_features": ";".join(feature_names[i] for i in top_idx)}
    print(f"  -> stability: Kendall's W (top-{len(top_idx)} across {REPEATS} repeats) = {w:.3f}")
    return out, stability


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exercise", help="e.g. Ex1")
    g.add_argument("--all", action="store_true", help="run all 6 REHAB24 exercises")
    p.add_argument("--family", default="anatomical", help="feature family: anatomical | biophases")
    args = p.parse_args()

    exercises = [f"Ex{i}" for i in range(1, 7)] if args.all else [args.exercise]

    results = [run_exercise(ex, args.family) for ex in exercises]
    frames = [r[0] for r in results]
    stability_rows = [r[1] for r in results]

    combined = pd.concat(frames, ignore_index=True)
    out_path = SHAP_DIR / f"rehab24_{args.family}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stability_path = SHAP_DIR / f"rehab24_{args.family}_stability.csv"
    stability_df = pd.DataFrame(stability_rows)

    if not args.all and out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["exercise"] != args.exercise]
        combined = pd.concat([existing, combined], ignore_index=True)

        if stability_path.exists():
            existing_stability = pd.read_csv(stability_path)
            existing_stability = existing_stability[existing_stability["exercise"] != args.exercise]
            stability_df = pd.concat([existing_stability, stability_df], ignore_index=True)

    combined.to_csv(out_path, index=False)
    stability_df.to_csv(stability_path, index=False)
    print(f"\n{combined.shape[0]} righe totali -> {out_path}")
    print(f"{stability_df.shape[0]} righe di stabilita' -> {stability_path}")


if __name__ == "__main__":
    main()
