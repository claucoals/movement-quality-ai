"""
Same kinematic features as build_features_ex5.py (knee/hip angle ROM, velocity,
tempo), computed on all 5 KIMORE exercises instead of only ex5, one row per
subject per exercise. cTS is the same per-subject clinical score regardless of
exercise, so this is a diagnostic: does the knee/hip signal correlate with cTS
in any exercise, or only (or nowhere) in the squat?

Note: rows from the same subject across exercises are not independent (same
person, same cTS) - fine for a univariate correlation check, but must be kept
in mind before using this for CV (needs grouping by subject, not by row).

Usage:
    python src/build_features_pooled.py
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "kimore_exercise_dataset.pkl"
OUT_PATH = ROOT / "data" / "features_pooled.csv"

ANGLE_DEFS = {
    "knee_left":  ("hipleft", "kneeleft", "ankleleft"),
    "knee_right": ("hipright", "kneeright", "ankleright"),
    "hip_left":   ("shoulderleft", "hipleft", "kneeleft"),
    "hip_right":  ("shoulderright", "hipright", "kneeright"),
}


def positions(row: pd.Series, joint: str) -> np.ndarray:
    return np.asarray(row[joint])[:, 4:7]


def angle_series(row: pd.Series, a: str, b: str, c: str) -> np.ndarray:
    pa, pb, pc = positions(row, a), positions(row, b), positions(row, c)
    ba, bc = pa - pb, pc - pb
    cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def rep_tempo_features(knee_angle: np.ndarray) -> dict:
    n = len(knee_angle)
    troughs, _ = find_peaks(-knee_angle, prominence=10, distance=max(n // 30, 5))
    if len(troughs) < 2:
        return {"n_reps": len(troughs), "rep_dur_mean": np.nan, "rep_dur_std": np.nan}
    rep_durations = np.diff(troughs) / n
    return {
        "n_reps": len(troughs),
        "rep_dur_mean": rep_durations.mean(),
        "rep_dur_std": rep_durations.std(),
    }


def subject_features(row: pd.Series) -> dict:
    feats = {}
    angle_signals = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        s = angle_series(row, a, b, c)
        angle_signals[name] = s
        v = np.diff(s)
        feats[f"{name}_min"] = s.min()
        feats[f"{name}_max"] = s.max()
        feats[f"{name}_rom"] = s.max() - s.min()
        feats[f"{name}_mean"] = s.mean()
        feats[f"{name}_std"] = s.std()
        feats[f"{name}_vel_mean_abs"] = np.abs(v).mean()
        feats[f"{name}_vel_std"] = v.std()
        feats[f"{name}_vel_max_abs"] = np.abs(v).max()
    feats["sym_knee"] = abs(feats["knee_left_rom"] - feats["knee_right_rom"])
    feats["sym_hip"] = abs(feats["hip_left_rom"] - feats["hip_right_rom"])

    knee_avg = (angle_signals["knee_left"] + angle_signals["knee_right"]) / 2
    feats.update(rep_tempo_features(knee_avg))
    return feats


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    all_rows = []
    for ex_name, ex_df in data.items():
        for subj_i, row in ex_df.iterrows():
            feats = subject_features(row)
            feats["exercise"] = ex_name
            feats["subject_id"] = f"{ex_name}_{subj_i}"
            feats["cTS"] = row["cTS"]
            all_rows.append(feats)

    out = pd.DataFrame(all_rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"{out.shape[0]} righe (soggetto x esercizio), {out.shape[1] - 3} feature + cTS -> {OUT_PATH}")
    print(out.groupby("exercise").size())


if __name__ == "__main__":
    main()
