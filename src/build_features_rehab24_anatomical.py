"""
Anatomical (named joint-angle) features for REHAB24-6, replacing the opaque trajectory-PCA
components in build_features_rehab24.py now that the joint mapping is confirmed
(data/raw/rehab24/joints_names.txt, downloaded from the dataset's Zenodo record - a
mocap-standard 26-joint hierarchy, verified anatomically plausible: Head_end < Neck < Spine1 <
Spine < Hips < knees < feet in image-y).

Mirrors the angle logic in build_features_ex5.py (angle at a joint from its two neighbours),
applied to every major joint so the same pipeline covers all 6 exercises (arm abduction, arm
VW, push-ups, leg abduction, leg lunge, squats) without hand-picking a different feature set
per exercise - which joints matter for which exercise is what SHAP/feature-importance reveals,
not something to decide upfront by looking at the exercise name.

Subject id comes from data/raw/rehab24/annotations.csv's person_id (see
rehab24_annotations.py), not the filename's "PM_NNN" string - two different filename strings
can be the same real person. Reps flagged `mocap_erroneous` there are excluded.

Usage:
    python src/build_features_rehab24_anatomical.py --exercise Ex1
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from anatomical_features import angle_at_vertex, knee_valgus_ratio, summary_features
from rehab24_common import iter_reps, parse_exercise_arg

from paths import rehab24_features

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
    return angle_at_vertex(arr[:, JOINT_IDX[a]], arr[:, JOINT_IDX[b]], arr[:, JOINT_IDX[c]])


def knee_valgus_proxy(arr: np.ndarray) -> np.ndarray:
    """REHAB24-specific joint lookup for the shared knee_valgus_ratio formula
    (anatomical_features.py) - "l_leg"/"r_leg" are this schema's knee joints (see JOINT_IDX;
    "leg" = shin bone, starting at the knee), "l_upleg"/"r_upleg" its hip joints. Uses x
    (lateral pixel) distance only, since the camera is roughly frontal for these exercises
    (confirmed via data/raw/rehab24/annotations.csv's `orientation` column - true for 5/6
    exercises, not Ex3/push-ups, which is filmed in profile; harmless there since valgus isn't
    a push-up quality marker and SHAP doesn't pick it up for that exercise)."""
    return knee_valgus_ratio(arr[:, JOINT_IDX["l_leg"]], arr[:, JOINT_IDX["r_leg"]],
                              arr[:, JOINT_IDX["l_upleg"]], arr[:, JOINT_IDX["r_upleg"]])


def subject_features(arr: np.ndarray) -> dict:
    angle_signals = {name: angle_series(arr, *triple) for name, triple in ANGLE_DEFS.items()}
    return summary_features(angle_signals, knee_valgus_proxy(arr))


def main():
    exercise = parse_exercise_arg()
    out_path = rehab24_features(exercise, "anatomical")

    rows = []
    for rep in iter_reps(exercise):
        feats = subject_features(rep.arr)
        feats["subject"] = rep.subject
        feats["correct"] = rep.correct
        rows.append(feats)

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    print(f"{out.shape[0]} ripetizioni, {out['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature anatomiche con nome -> {out_path}")


if __name__ == "__main__":
    main()
