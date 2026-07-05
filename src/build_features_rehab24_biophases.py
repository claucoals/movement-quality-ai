"""
Biomechanically-segmented anatomical features for REHAB24-6.

build_features_rehab24_phases.py split every rep into three equal *time* thirds - and that
didn't help (notebook 07): the boundaries were arbitrary relative to the actual movement.
This instead finds each rep's own turnaround point from its primary movement angle - the
same per-exercise joint visually validated in notebook 06's EDA (shoulder for arm abduction,
elbow for arm-VW/push-ups, hip for leg abduction, knee for lunge/squat) - and splits into
descent / bottom / ascent around that event, so the boundary tracks the movement itself
rather than the clock.

Which *side* (left/right) to read that joint from is not fixed: data/raw/rehab24/annotations.csv
(exercise_subtype) and the filename's own variant field show Ex1 is 100% "rightarm", and Ex4/Ex5
mix left/right variants - a fixed side would silently track the wrong limb for roughly half of
those reps (see src/rehab24_annotations.py). Ex2/Ex3/Ex6 have no variant (bilateral movements),
so a fixed side is harmless there.

Within each of the three phases it computes the same anatomical angle features as
build_features_rehab24_anatomical.py (min/max/rom/mean/std/velocity per joint angle,
left/right symmetry, knee-valgus proxy) - phase-localized versions of features SHAP
(notebook 08) already showed carry real, clinically-interpretable signal, rather than
opaque per-phase PCA components. rep_tempo_features is dropped here: it detects rep
boundaries via find_peaks and doesn't make sense on a sub-phase slice of a single rep.

Usage:
    python src/build_features_rehab24_biophases.py --exercise Ex1
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd

from build_features_rehab24_anatomical import ANGLE_DEFS, angle_series, knee_valgus_proxy
from rehab24_annotations import is_mocap_erroneous, side_for, subject_id_for

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "rehab24"

FILENAME_RE = re.compile(r"^(PM_\w+)_c(\d+)_(.+)-rep(\d+)-(\d+)\.npy$")

# Joint type used for phase-turnaround detection, per exercise; combined with side_for()'s
# left/right call (from the rep's own variant) to pick the actual ANGLE_DEFS key.
JOINT_TYPE = {
    "Ex1": "shoulder_angle",  # arm abduction
    "Ex2": "elbow",           # arm VW
    "Ex3": "elbow",           # push-ups
    "Ex4": "hip",             # leg abduction
    "Ex5": "knee",            # lunge
    "Ex6": "knee",            # squat
}

PHASES = ("descent", "bottom", "ascent")


def reference_angle_name(exercise: str, variant: str) -> str:
    return f"{side_for(exercise, variant)}_{JOINT_TYPE[exercise]}"


def find_phase_bounds(signal: np.ndarray) -> tuple[int, int]:
    """Locate the rep's turnaround from its own reference-angle signal: whichever extremum
    (min or max) is more displaced from the trial's start/end baseline, so this works whether
    the exercise's angle increases (arm abduction) or decreases (squat knee flexion) during
    the active part of the rep. Returns a fixed-width window (bottom_start, bottom_end)
    centered on that turnaround; descent is everything before it, ascent everything after."""
    n = len(signal)
    edge = max(n // 10, 1)
    baseline = (signal[:edge].mean() + signal[-edge:].mean()) / 2
    idx_min, idx_max = int(np.argmin(signal)), int(np.argmax(signal))
    turnaround = idx_min if abs(signal[idx_min] - baseline) > abs(signal[idx_max] - baseline) else idx_max
    half_width = max(int(0.075 * n), 2)
    lo = max(turnaround - half_width, 1)
    hi = min(turnaround + half_width, n - 1)
    if hi <= lo:
        lo, hi = n // 3, 2 * n // 3  # degenerate (very short rep): fall back to thirds
    return lo, hi


def phase_features(arr_phase: np.ndarray, prefix: str) -> dict:
    feats = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        s = angle_series(arr_phase, a, b, c)
        v = np.diff(s)
        feats[f"{prefix}__{name}_min"] = s.min()
        feats[f"{prefix}__{name}_max"] = s.max()
        feats[f"{prefix}__{name}_rom"] = s.max() - s.min()
        feats[f"{prefix}__{name}_mean"] = s.mean()
        feats[f"{prefix}__{name}_std"] = s.std()
        feats[f"{prefix}__{name}_vel_mean_abs"] = np.abs(v).mean() if len(v) else 0.0

    feats[f"{prefix}__sym_elbow"] = abs(feats[f"{prefix}__l_elbow_rom"] - feats[f"{prefix}__r_elbow_rom"])
    feats[f"{prefix}__sym_knee"] = abs(feats[f"{prefix}__l_knee_rom"] - feats[f"{prefix}__r_knee_rom"])
    feats[f"{prefix}__sym_hip"] = abs(feats[f"{prefix}__l_hip_rom"] - feats[f"{prefix}__r_hip_rom"])
    feats[f"{prefix}__sym_shoulder"] = abs(feats[f"{prefix}__l_shoulder_angle_rom"]
                                            - feats[f"{prefix}__r_shoulder_angle_rom"])

    valgus = knee_valgus_proxy(arr_phase)
    feats[f"{prefix}__knee_valgus_min"] = valgus.min()
    feats[f"{prefix}__knee_valgus_mean"] = valgus.mean()
    return feats


def subject_features(arr: np.ndarray, exercise: str, variant: str) -> dict:
    ref_name = reference_angle_name(exercise, variant)
    ref_signal = angle_series(arr, *ANGLE_DEFS[ref_name])
    lo, hi = find_phase_bounds(ref_signal)
    bounds = {"descent": (0, lo), "bottom": (lo, hi), "ascent": (hi, len(arr))}

    feats = {}
    for phase in PHASES:
        a, b = bounds[phase]
        feats.update(phase_features(arr[a:b], phase))
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exercise", required=True, help="e.g. Ex1")
    args = p.parse_args()

    ex_dir = BASE_DIR / f"{args.exercise}-segmented"
    out_path = ROOT / "data" / f"features_rehab24_{args.exercise.lower()}_biophases.csv"

    rows = []
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
        arr = np.load(f)
        feats = subject_features(arr, args.exercise, variant)
        feats["subject"] = subject_id_for(f.name)
        feats["correct"] = int(label)
        rows.append(feats)

    print(f"{len(rows)} ripetizioni caricate, {n_skipped} file scartati (nome non conforme), "
          f"{n_mocap_erroneous} scartati (mocap_erroneous)")
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"{out.shape[0]} ripetizioni, {out['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature ({len(PHASES)} fasi biomeccaniche) -> {out_path}")


if __name__ == "__main__":
    main()
