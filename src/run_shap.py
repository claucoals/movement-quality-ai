"""
Out-of-fold SHAP values for REHAB24-6's anatomical feature family.

For each exercise this picks whichever model type actually won that exercise in
results/experiments.csv (read, not hardcoded - if a rerun of the sweep changes the winner,
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
    python src/run_shap.py --exercise Ex1
    python src/run_shap.py --all
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import GridSearchCV

from quality_model import build_models, _outer_splits

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


def pick_winning_model(exercise: str) -> str:
    res = pd.read_csv(ROOT / "results" / "experiments.csv")
    sub = res[(res["dataset"] == f"rehab24_{exercise.lower()}_anatomical") & (res["model"] != "dummy")]
    if sub.empty:
        raise SystemExit(f"No results for rehab24_{exercise.lower()}_anatomical in results/experiments.csv - "
                          f"run src/run_experiments.py first.")
    means = sub.groupby("model")["auc"].mean()
    return means.idxmax()


def run_exercise(exercise: str) -> pd.DataFrame:
    path = ROOT / "data" / f"features_rehab24_{exercise.lower()}_anatomical.csv"
    df = pd.read_csv(path)
    y = df["correct"]
    groups = df["subject"]
    X = df.drop(columns=["correct", "subject"]).select_dtypes(include=[np.number]).reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)
    feature_names = X.columns.tolist()
    n = len(X)

    model_name = pick_winning_model(exercise)
    print(f"[{exercise}] {n} samples, {groups.nunique()} subjects, winning model: {model_name}")

    pipe_template, grid = build_models("classification", SEED)[model_name]

    shap_sum = np.zeros((n, len(feature_names)))
    shap_count = np.zeros(n)
    n_folds_done = 0

    for repeat_i, fold_i, tr, te in _outer_splits(X, y, groups, "classification", OUTER, REPEATS, SEED):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr]
        groups_tr = groups.iloc[tr]

        inner_splits = list(_outer_splits(Xtr, ytr, groups_tr, "classification", INNER, 1, SEED + repeat_i))
        inner_cv = [(a, b) for _, _, a, b in inner_splits]

        gs = GridSearchCV(pipe_template, grid, scoring="balanced_accuracy", cv=inner_cv, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best_pipe = gs.best_estimator_

        def predict_fn(arr, _pipe=best_pipe):
            return _pipe.predict_proba(pd.DataFrame(arr, columns=feature_names))[:, 1]

        masker = shap.maskers.Independent(Xtr.values, max_samples=Xtr.shape[0])
        explainer = shap.Explainer(predict_fn, masker, feature_names=feature_names)
        sv = explainer(Xte.values)

        shap_sum[te] += sv.values
        shap_count[te] += 1
        n_folds_done += 1
        print(f"  repeat {repeat_i + 1} fold {fold_i + 1}: best_params={gs.best_params_}, "
              f"explained {len(te)} held-out samples")

    assert (shap_count == REPEATS).all(), "every sample must be held out exactly once per repeat"
    mean_shap = shap_sum / shap_count[:, None]

    out = pd.DataFrame(mean_shap, columns=[f"shap__{c}" for c in feature_names])
    out.insert(0, "exercise", exercise)
    out.insert(1, "subject", groups.values)
    out.insert(2, "correct", y.values)
    out.insert(3, "model", model_name)
    for c in feature_names:
        out[f"value__{c}"] = X[c].values
    print(f"  -> {n} samples explained ({n_folds_done} fold fits total)")
    return out


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--exercise", help="e.g. Ex1")
    g.add_argument("--all", action="store_true", help="run all 6 REHAB24 exercises")
    args = p.parse_args()

    exercises = [f"Ex{i}" for i in range(1, 7)] if args.all else [args.exercise]

    frames = [run_exercise(ex) for ex in exercises]
    combined = pd.concat(frames, ignore_index=True)
    out_path = ROOT / "results" / "shap_rehab24_anatomical.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.all and out_path.exists():
        existing = pd.read_csv(out_path)
        existing = existing[existing["exercise"] != args.exercise]
        combined = pd.concat([existing, combined], ignore_index=True)

    combined.to_csv(out_path, index=False)
    print(f"\n{combined.shape[0]} righe totali -> {out_path}")


if __name__ == "__main__":
    main()
