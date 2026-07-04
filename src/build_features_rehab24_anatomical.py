"""
Anatomical (named joint-angle) features for REHAB24-6 - the interpretable feature family
the project's thesis actually calls for, replacing the opaque trajectory-PCA components in
build_features_rehab24.py now that the joint mapping is confirmed (data/ui_prmd/joints_names.txt,
downloaded from the dataset's Zenodo record - a mocap-standard 26-joint hierarchy, verified
anatomically plausible: Head_end < Neck < Spine1 < Spine < Hips < knees < feet in image-y).

Mirrors the angle logic in build_features_ex5.py (angle at a joint from its two neighbours),
applied to every major joint so the same pipeline covers all 6 exercises (arm abduction, arm
VW, push-ups, leg abduction, leg lunge, squats) without hand-picking a different feature set
per exercise - which joints matter for which exercise is exactly what SHAP/feature-importance
should reveal later (Fase 4), not something to decide upfront by looking at the exercise name.

Usage:
    python src/build_features_rehab24_anatomical.py --exercise Ex1
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "ui_prmd" / "2d_joints_segmented" / "2d_joints_segmented"

FILENAME_RE = re.compile(r"^(PM_\d+)_c(\d+)_(.+)-rep(\d+)-(\d+)\.npy$")

JOINT_IDX = {
    "hips": 0, "spine": 1, "spine1": 2, "neck": 3, "head": 4, "head_end": 5,
    "l_shoulder": 6, "l_arm": 7, "l_forearm": 8, "l_hand": 9, "l_hand_end": 10,
    "r_shoulder": 11, "r_arm": 12, "r_forearm": 13, "r_hand": 14, "r_hand_end": 15,
    "l_upleg": 16, "l_leg": 17, "l_foot": 18, "l_toe": 19, "l_toe_end": 20,
    "r_upleg": 21, "r_leg": 22, "r_foot": 23, "r_toe": 24, "r_toe_end": 25,
}

# angle at the middle joint (vertex), formed by the two flanking joints
ANGLE_DEFS = {
    "l_elbow": ("l_arm", "l_forearm", "l_hand"),
    "r_elbow": ("r_arm", "r_forearm", "r_hand"),
    "l_shoulder_angle": ("neck", "l_shoulder", "l_arm"),
    "r_shoulder_angle": ("neck", "r_shoulder", "r_arm"),
    "l_knee": ("l_upleg", "l_leg", "l_foot"),
    "r_knee": ("r_upleg", "r_leg", "r_foot"),
    "l_hip": ("spine1", "l_upleg", "l_leg"),
    "r_hip": ("spine1", "r_upleg", "r_leg"),
    "l_ankle": ("l_leg", "l_foot", "l_toe"),
    "r_ankle": ("r_leg", "r_foot", "r_toe"),
    "trunk_flex": ("neck", "spine1", "hips"),
}


def angle_series(arr: np.ndarray, a: str, b: str, c: str) -> np.ndarray:
    pa, pb, pc = arr[:, JOINT_IDX[a]], arr[:, JOINT_IDX[b]], arr[:, JOINT_IDX[c]]
    ba, bc = pa - pb, pc - pb
    cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def knee_valgus_proxy(arr: np.ndarray) -> np.ndarray:
    """Ratio of inter-knee to inter-hip distance: well below 1 means the knees are caving
    in relative to the hips (valgus), a classic squat/lunge quality fault. Uses x (lateral
    pixel) distance only, since the camera is roughly frontal for these exercises."""
    knee_dist = np.abs(arr[:, JOINT_IDX["l_leg"]][:, 0] - arr[:, JOINT_IDX["r_leg"]][:, 0])
    hip_dist = np.abs(arr[:, JOINT_IDX["l_upleg"]][:, 0] - arr[:, JOINT_IDX["r_upleg"]][:, 0])
    return knee_dist / (hip_dist + 1e-6)


def rep_tempo_features(signal: np.ndarray) -> dict:
    n = len(signal)
    troughs, _ = find_peaks(-signal, prominence=5, distance=max(n // 10, 3))
    if len(troughs) < 2:
        return {"n_reps": len(troughs), "rep_dur_mean": np.nan, "rep_dur_std": np.nan}
    durations = np.diff(troughs) / n
    return {"n_reps": len(troughs), "rep_dur_mean": durations.mean(), "rep_dur_std": durations.std()}


def subject_features(arr: np.ndarray) -> dict:
    feats = {}
    angle_signals = {}
    for name, (a, b, c) in ANGLE_DEFS.items():
        s = angle_series(arr, a, b, c)
        angle_signals[name] = s
        v = np.diff(s)
        feats[f"{name}_min"] = s.min()
        feats[f"{name}_max"] = s.max()
        feats[f"{name}_rom"] = s.max() - s.min()
        feats[f"{name}_mean"] = s.mean()
        feats[f"{name}_std"] = s.std()
        feats[f"{name}_vel_mean_abs"] = np.abs(v).mean()
        feats[f"{name}_vel_max_abs"] = np.abs(v).max()

    feats["sym_elbow"] = abs(feats["l_elbow_rom"] - feats["r_elbow_rom"])
    feats["sym_knee"] = abs(feats["l_knee_rom"] - feats["r_knee_rom"])
    feats["sym_hip"] = abs(feats["l_hip_rom"] - feats["r_hip_rom"])
    feats["sym_shoulder"] = abs(feats["l_shoulder_angle_rom"] - feats["r_shoulder_angle_rom"])

    valgus = knee_valgus_proxy(arr)
    feats["knee_valgus_min"] = valgus.min()
    feats["knee_valgus_mean"] = valgus.mean()

    knee_avg = (angle_signals["l_knee"] + angle_signals["r_knee"]) / 2
    feats.update(rep_tempo_features(knee_avg))
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exercise", required=True, help="e.g. Ex1")
    args = p.parse_args()

    ex_dir = BASE_DIR / f"{args.exercise}-segmented"
    out_path = ROOT / "data" / f"features_rehab24_{args.exercise.lower()}_anatomical.csv"

    rows = []
    n_skipped = 0
    for f in sorted(ex_dir.iterdir()):
        if f.suffix != ".npy":
            continue
        m = FILENAME_RE.match(f.name)
        if not m:
            n_skipped += 1
            continue
        subject, cam, variant, rep, label = m.groups()
        arr = np.load(f)
        feats = subject_features(arr)
        feats["subject"] = subject
        feats["correct"] = int(label)
        rows.append(feats)

    print(f"{len(rows)} ripetizioni caricate, {n_skipped} file scartati (nome non conforme)")
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"{out.shape[0]} ripetizioni, {out['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature anatomiche con nome -> {out_path}")


if __name__ == "__main__":
    main()
