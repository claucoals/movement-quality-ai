"""
Main experiment runner: reads config.yaml, runs nested CV (quality_model.nested_cv) on
every dataset listed there, and appends every fold's result to one CSV.

This is the only place that decides *what* gets run (via config.yaml) and *where* results
go (results/experiments.csv). Notebooks only read that CSV - they never hold hand-typed
numbers, so a rerun with a changed config always stays consistent with what notebooks show.

Usage:
    python src/run_experiments.py
    python src/run_experiments.py --config config.yaml
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from quality_model import nested_cv, results_to_frame

ROOT = Path(__file__).resolve().parents[1]


def run_dataset(entry: dict, cv_cfg: dict, seed: int) -> pd.DataFrame:
    path = ROOT / entry["features"]
    df = pd.read_csv(path)
    y = df[entry["target"]]
    groups_col = entry.get("groups")
    groups = df[groups_col] if groups_col else None
    drop_cols = [entry["target"]] + ([groups_col] if groups_col else [])
    X = df.drop(columns=drop_cols).select_dtypes(include=[np.number])
    repeats = entry.get("repeats", cv_cfg["repeats"])

    print(f"[{entry['name']}] {X.shape[0]} samples, {X.shape[1]} features, task={entry['task']}, "
          f"repeats={repeats}"
          + (f", groups={groups_col} ({groups.nunique()})" if groups is not None else ", no groups"))

    results = nested_cv(
        X, y, task=entry["task"], seed=seed,
        outer=cv_cfg["outer_splits"], inner=cv_cfg["inner_splits"], repeats=repeats,
        groups=groups, verbose=False,
    )
    return results_to_frame(
        results, dataset=entry["name"], task=entry["task"],
        n_samples=X.shape[0], n_features=X.shape[1],
        grouped=groups is not None, status=entry.get("status", "active"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--only", default=None,
                    help="comma-separated dataset names to (re)run, e.g. "
                         "rehab24_ex1_dynamics,rehab24_ex2_dynamics - skips the rest. "
                         "Existing rows for those names in the output CSV are replaced, "
                         "everything else in the CSV is left untouched.")
    args = p.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    cv_cfg = {"outer_splits": cfg["outer_splits"], "inner_splits": cfg["inner_splits"],
              "repeats": cfg["repeats"]}

    datasets = cfg["datasets"]
    if args.only:
        wanted = set(args.only.split(","))
        datasets = [e for e in datasets if e["name"] in wanted]
        missing = wanted - {e["name"] for e in datasets}
        if missing:
            raise SystemExit(f"Unknown dataset name(s) in --only: {missing}")

    all_frames = []
    for entry in datasets:
        frame = run_dataset(entry, cv_cfg, cfg["seed"])
        all_frames.append(frame)
        n_combos = frame.shape[0]
        print(f"  -> {n_combos} righe (modelli x repeat x fold)")

    new_results = pd.concat(all_frames, ignore_index=True)
    out_path = ROOT / cfg["output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        existing = pd.read_csv(out_path)
        ran_names = new_results["dataset"].unique()
        existing = existing[~existing["dataset"].isin(ran_names)]
        combined = pd.concat([existing, new_results], ignore_index=True)
    else:
        combined = new_results

    combined.to_csv(out_path, index=False)
    print(f"\n{new_results.shape[0]} righe nuove/aggiornate, {combined.shape[0]} righe totali -> {out_path}")


if __name__ == "__main__":
    main()
