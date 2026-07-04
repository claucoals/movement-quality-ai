"""
Movement-smoothness features for KIMORE ex5 (squat), one row per subject.

Every earlier feature family used position/angle and its first derivative
(velocity). Smoothness of movement (how jerky vs fluid the execution is) is a
separate, well-established clinical marker of motor control quality, distinct
from ROM or tempo: it needs the second derivative (angular acceleration) and
third (jerk). Computed on the same knee/hip angle signals as
build_features_ex5.py, so directly comparable/combinable with that set.

Usage:
    python src/build_features_smoothness.py
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "kimore_exercise_dataset.pkl"
OUT_PATH = ROOT / "data" / "features_smoothness_ex5.csv"

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


def smoothness_features(angle: np.ndarray) -> dict:
    """Normalized jerk (Hogan & Sternad): third derivative of the signal,
    normalized by duration and amplitude so subjects with different recording
    lengths or ROM are comparable. Lower = smoother/more controlled movement."""
    vel = np.diff(angle)
    acc = np.diff(vel)
    jerk = np.diff(acc)
    duration = len(angle)
    amplitude = angle.max() - angle.min()
    if amplitude < 1e-6:
        norm_jerk = np.nan
    else:
        mean_sq_jerk = np.mean(jerk ** 2)
        norm_jerk = np.sqrt(0.5 * mean_sq_jerk * duration ** 5) / amplitude
    return {
        "jerk_mean_abs": np.abs(jerk).mean(),
        "jerk_std": jerk.std(),
        "acc_mean_abs": np.abs(acc).mean(),
        "acc_std": acc.std(),
        "normalized_jerk": norm_jerk,
    }


def subject_features(row: pd.Series) -> dict:
    feats = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        s = angle_series(row, a, b, c)
        for k, v in smoothness_features(s).items():
            feats[f"{name}_{k}"] = v
    return feats


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    ex5 = data["ex5"]

    rows = [subject_features(row) for _, row in ex5.iterrows()]
    out = pd.DataFrame(rows)
    out["cTS"] = ex5["cTS"].values

    out.to_csv(OUT_PATH, index=False)
    print(f"{out.shape[0]} soggetti, {out.shape[1] - 1} feature + target -> {OUT_PATH}")
    print(out.describe().T[["min", "mean", "max"]])


if __name__ == "__main__":
    main()
