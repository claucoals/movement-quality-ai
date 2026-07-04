"""
Kinematic features for KIMORE ex5 (squat), one row per subject.

Mirrors the angle logic in pose_to_features.py (angle at a joint from the two
adjacent joints), but reads KIMORE's Kinect skeleton arrays instead of
MediaPipe landmarks. Each joint cell is an array (n_frames, 7) =
[qx, qy, qz, qw, x, y, z]; only the xyz position (columns 4:7) is used.

Usage:
    python src/build_features_ex5.py
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "kimore_exercise_dataset.pkl"
OUT_PATH = ROOT / "data" / "features_ex5.csv"

# angle is measured at the middle joint (vertex), lower-body only: relevant for a squat
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


def subject_features(row: pd.Series) -> dict:
    feats = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        s = angle_series(row, a, b, c)
        feats[f"{name}_min"] = s.min()
        feats[f"{name}_max"] = s.max()
        feats[f"{name}_rom"] = s.max() - s.min()
        feats[f"{name}_mean"] = s.mean()
        feats[f"{name}_std"] = s.std()
    feats["sym_knee"] = abs(feats["knee_left_rom"] - feats["knee_right_rom"])
    feats["sym_hip"] = abs(feats["hip_left_rom"] - feats["hip_right_rom"])
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
