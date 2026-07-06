"""
Runs ONE experiment - nested CV (quality_model.nested_cv) on the single dataset described
in config.yaml's `dataset:` block - and appends its rows to results/experiments/experiments.csv
(or wherever config.yaml's `output` points).

One dataset per launch, on purpose: to run a different one, copy its fields from
datasets.yaml into config.yaml's `dataset:` block and rerun. Notebooks only read the output
CSV - they never hold hand-typed numbers, so results accumulate across separate, deliberate
launches instead of one long unattended sweep.

Usage:
    python src/run_experiments.py
    python src/run_experiments.py --config config.yaml
"""

from __future__ import annotations
import argparse
import time
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
    outer = entry.get("outer_splits", cv_cfg["outer_splits"])

    print(f"[{entry['name']}] {X.shape[0]} samples, {X.shape[1]} features, task={entry['task']}, "
          f"outer={outer}, repeats={repeats}"
          + (f", groups={groups_col} ({groups.nunique()})" if groups is not None else ", no groups"),
          flush=True)

    results = nested_cv(
        X, y, task=entry["task"], seed=seed,
        outer=outer, inner=cv_cfg["inner_splits"], repeats=repeats,
        groups=groups, verbose=False,
    )
    return results_to_frame(
        results, dataset=entry["name"], task=entry["task"],
        n_samples=X.shape[0], n_features=X.shape[1],
        grouped=groups is not None, status=entry.get("status", "active"),
    )


def save_results(new_results: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Merge new_results into out_path, replacing any existing rows for the same dataset
    names, and write immediately - called after every dataset, not just once at the end."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        existing = pd.read_csv(out_path)
        ran_names = new_results["dataset"].unique().tolist()
        existing = existing.loc[~existing["dataset"].isin(ran_names)]
        combined = pd.concat([existing, new_results], ignore_index=True)
    else:
        combined = new_results
    combined.to_csv(out_path, index=False)
    return combined


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    with open(ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    cv_cfg = {"outer_splits": cfg["outer_splits"], "inner_splits": cfg["inner_splits"],
              "repeats": cfg["repeats"]}
    out_path = ROOT / cfg["output"]

    t_start = time.monotonic()
    frame = run_dataset(cfg["dataset"], cv_cfg, cfg["seed"])
    save_results(frame, out_path)

    print(f"\n{frame.shape[0]} righe (modelli x repeat x fold) in "
          f"{time.monotonic() - t_start:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
