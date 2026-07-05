"""
Main experiment runner: reads config.yaml, runs nested CV (quality_model.nested_cv) on
every dataset listed there, and appends every fold's result to one CSV.

This is the only place that decides *what* gets run (via config.yaml) and *where* results
go (results/experiments/experiments.csv). Notebooks only read that CSV - they never hold hand-typed
numbers, so a rerun with a changed config always stays consistent with what notebooks show.

Progress is printed with flush=True and a timestamp after every dataset (not buffered until
the whole run ends, which used to make background runs look silent for hours), and results
are written to disk after each dataset too, not only once at the very end - a sweep killed
partway through (which has happened more than once this session, chasing down a bug found
mid-run) now only loses the one dataset in flight, not every dataset already computed.

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
    p.add_argument("--only", default=None,
                    help="comma-separated dataset names to (re)run, e.g. "
                         "rehab24_ex1_dynamics,rehab24_ex2_dynamics - skips the rest. "
                         "Existing rows for those names in the output CSV are replaced, "
                         "everything else in the CSV is left untouched.")
    p.add_argument("--output", default=None,
                    help="override config.yaml's output path - use a different file when "
                         "running concurrently with another run_experiments.py process, to "
                         "avoid both writing to results/experiments/experiments.csv at once.")
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

    out_path = ROOT / (args.output or cfg["output"])
    n_total = len(datasets)
    t_start = time.monotonic()
    all_new_rows = 0

    for i, entry in enumerate(datasets, 1):
        t_dataset_start = time.monotonic()
        frame = run_dataset(entry, cv_cfg, cfg["seed"])
        combined = save_results(frame, out_path)
        all_new_rows += frame.shape[0]

        elapsed = time.monotonic() - t_start
        per_dataset = elapsed / i
        eta = per_dataset * (n_total - i)
        print(f"  -> {frame.shape[0]} righe (modelli x repeat x fold) in "
              f"{time.monotonic() - t_dataset_start:.0f}s. "
              f"[{i}/{n_total}] fatti, {elapsed / 60:.1f} min trascorsi, "
              f"~{eta / 60:.1f} min stimati alla fine -> {out_path}",
              flush=True)

    print(f"\n{all_new_rows} righe nuove/aggiornate totali, "
          f"{time.monotonic() - t_start:.0f}s totali -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
