"""
Reference trajectories and deviation-from-reference (CoRe-light, "Pipeline B") for REHAB24-6.

SHAP on biophases (Pipeline A) answers "which components does the model rely on". This
answers a different question: "where does a given rep's own movement differ from correct
technique, joint by joint, phase by phase" - the other half needed for the Movement
Attribution Map (SHAP importance x reference deviation).

For each exercise, each of the 11 named joint angles plus the knee-valgus proxy, each of the
3 biomechanical phases (descent/bottom/ascent - same turnaround-based boundaries as
build_features_rehab24_biophases.py, reused here rather than redefined):
  1. slice that phase out of every rep's angle signal
  2. resample it to a fixed length (phases have different frame counts across reps, since the
     turnaround point falls at a different frame each time)
  3. average the resampled phase-trajectories of *correct* reps only -> the reference
  4. every rep's deviation = mean absolute difference between its own resampled
     phase-trajectory and that reference

This is deliberately a descriptive/interpretability artifact, not a new classifier input: the
reference is built from all correct reps, including - for a given query rep - reps from its
own subject. That's the right choice for explaining a rep's deviation (more reps = a more
reliable reference), but if "deviation from reference" ever becomes a model feature instead of
an explanation, the reference would need to be rebuilt leave-subject-out first, the same
anti-leakage rule enforced everywhere else in this project (quality_model.py's grouped CV).

Usage:
    python src/build_reference_deviation.py --exercise Ex1
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

from build_features_rehab24_anatomical import ANGLE_DEFS, angle_series, knee_valgus_proxy
from build_features_rehab24_biophases import PHASES, find_phase_bounds, reference_angle_name
from rehab24_annotations import is_mocap_erroneous, subject_id_for

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "rehab24"
FILENAME_RE = re.compile(r"^(PM_\w+)_c(\d+)_(.+)-rep(\d+)-(\d+)\.npy$")

RESAMPLE_N = 20  # points per phase after resampling - a fixed length is what makes averaging
                 # and differencing trajectories across reps of different durations possible


def resample(signal: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    if len(signal) == 1:
        return np.full(n, signal[0])
    t_old = np.linspace(0.0, 1.0, len(signal))
    t_new = np.linspace(0.0, 1.0, n)
    return np.interp(t_new, t_old, signal)


def load_reps(exercise: str) -> tuple[list[dict], int, int]:
    ex_dir = BASE_DIR / f"{exercise}-segmented"
    reps = []
    n_skipped, n_mocap_erroneous = 0, 0
    for f in sorted(ex_dir.iterdir()):
        if f.suffix != ".npy":
            continue
        m = FILENAME_RE.match(f.name)
        if not m:
            n_skipped += 1
            continue
        if is_mocap_erroneous(f.name):
            n_mocap_erroneous += 1
            continue
        _, cam, variant, rep, label = m.groups()
        reps.append({"subject": subject_id_for(f.name), "variant": variant, "rep": int(rep),
                     "correct": int(label), "arr": np.load(f)})
    return reps, n_skipped, n_mocap_erroneous


def phase_trajectories(arr: np.ndarray, exercise: str, variant: str) -> dict:
    """Returns {(phase, angle_name): resampled trajectory} for one rep, angle_name ranging
    over every named joint angle plus 'knee_valgus'."""
    ref_signal = angle_series(arr, *ANGLE_DEFS[reference_angle_name(exercise, variant)])
    lo, hi = find_phase_bounds(ref_signal)
    bounds = {"descent": (0, lo), "bottom": (lo, hi), "ascent": (hi, len(arr))}

    out = {}
    for phase in PHASES:
        a, b = bounds[phase]
        sub_arr = arr[a:b]
        for angle_name, (ja, jb, jc) in ANGLE_DEFS.items():
            out[(phase, angle_name)] = resample(angle_series(sub_arr, ja, jb, jc))
        out[(phase, "knee_valgus")] = resample(knee_valgus_proxy(sub_arr))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exercise", required=True, help="e.g. Ex1")
    args = p.parse_args()
    exercise = args.exercise

    reps, n_skipped, n_mocap_erroneous = load_reps(exercise)
    print(f"{len(reps)} ripetizioni caricate, {n_skipped} file scartati (nome non conforme), "
          f"{n_mocap_erroneous} scartati (mocap_erroneous)")

    for rep in reps:
        rep["traj"] = phase_trajectories(rep["arr"], exercise, rep["variant"])

    correct_reps = [r for r in reps if r["correct"] == 1]
    print(f"{len(correct_reps)} ripetizioni corrette usate come riferimento "
          f"(su {len(reps)} totali)")

    angle_names = list(ANGLE_DEFS.keys()) + ["knee_valgus"]
    reference = {}
    for phase in PHASES:
        for angle_name in angle_names:
            key = (phase, angle_name)
            stacked = np.stack([r["traj"][key] for r in correct_reps])
            reference[key] = stacked.mean(axis=0)

    rows = []
    for rep in reps:
        row = {"subject": rep["subject"], "correct": rep["correct"]}
        for phase in PHASES:
            for angle_name in angle_names:
                key = (phase, angle_name)
                row[f"deviation__{phase}__{angle_name}"] = np.abs(rep["traj"][key] - reference[key]).mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = ROOT / "data" / f"features_rehab24_{exercise.lower()}_deviation.csv"
    out.to_csv(out_path, index=False)
    print(f"{out.shape[0]} ripetizioni, {out['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature di deviazione ({len(PHASES)} fasi x {len(angle_names)} "
          f"articolazioni) -> {out_path}")


if __name__ == "__main__":
    main()
