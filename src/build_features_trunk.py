"""
Trunk/pelvis kinematic features for KIMORE ex5 (squat), one row per subject.

KIMORE targets low-back-pain rehabilitation (its 5 exercises are arm lift,
lateral trunk tilt, trunk rotation, pelvis rotation, squat), so the clinically
relevant impairment is trunk/spine control, not knee/hip ROM per se. A classic
sign of poor squat quality in this population is compensating with the back
(excess trunk flexion or lateral lean) instead of moving through the hips.
Every earlier feature family (build_features_ex5.py, build_features_pooled.py,
build_features_trajectory.py) used only knee/hip angles - this fills that gap.

Uses Y as the vertical axis (verified: head Y > spinebase Y), X as lateral,
Z as depth, confirmed empirically on this pkl.

Usage:
    python src/build_features_trunk.py
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "kimore_exercise_dataset.pkl"
OUT_PATH = ROOT / "data" / "features_trunk_ex5.csv"


def positions(row: pd.Series, joint: str) -> np.ndarray:
    return np.asarray(row[joint])[:, 4:7]


def vector_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    cos = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def trunk_signals(row: pd.Series) -> dict:
    spinebase = positions(row, "spinebase")
    spinemid = positions(row, "spinemid")
    spineshoulder = positions(row, "spineshoulder")
    hipl, hipr = positions(row, "hipleft"), positions(row, "hipright")
    shol, shor = positions(row, "shoulderleft"), positions(row, "shoulderright")

    vertical = np.zeros_like(spinebase)
    vertical[:, 1] = 1.0  # Y is vertical (up)

    trunk_vec = spineshoulder - spinebase
    trunk_flex = vector_angle(trunk_vec, vertical)  # 0 = upright, larger = bent

    # lateral lean: trunk vector projected onto the frontal (X-Y) plane, angle from vertical
    trunk_frontal = trunk_vec.copy()
    trunk_frontal[:, 2] = 0.0
    vertical_frontal = vertical.copy()
    trunk_lateral_lean = vector_angle(trunk_frontal, vertical_frontal)

    # axial trunk rotation: angle between shoulder line and hip line, projected top-down (X-Z plane)
    shoulder_line = shor - shol
    hip_line = hipr - hipl
    shoulder_line_xz = shoulder_line.copy()
    shoulder_line_xz[:, 1] = 0.0
    hip_line_xz = hip_line.copy()
    hip_line_xz[:, 1] = 0.0
    trunk_rotation = vector_angle(shoulder_line_xz, hip_line_xz)

    return {
        "trunk_flex": trunk_flex,
        "trunk_lateral_lean": trunk_lateral_lean,
        "trunk_rotation": trunk_rotation,
    }


def subject_features(row: pd.Series) -> dict:
    feats = {}
    for name, s in trunk_signals(row).items():
        v = np.diff(s)
        feats[f"{name}_min"] = s.min()
        feats[f"{name}_max"] = s.max()
        feats[f"{name}_rom"] = s.max() - s.min()
        feats[f"{name}_mean"] = s.mean()
        feats[f"{name}_std"] = s.std()
        feats[f"{name}_vel_mean_abs"] = np.abs(v).mean()
        feats[f"{name}_vel_std"] = v.std()
        feats[f"{name}_vel_max_abs"] = np.abs(v).max()
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
