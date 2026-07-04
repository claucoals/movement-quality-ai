"""
video -> pose landmarks -> joint-angle time series -> summary kinematic features.

Front-end uses MediaPipe Pose (fast, runs on a laptop, no GPU needed). You can swap in
MMPose later for higher accuracy. The output features feed directly into quality_model.py.

Usage:
    python pose_to_features.py --video path/to/clip.mp4 --out features.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

try:
    import cv2
    import mediapipe as mp
    _POSE = mp.solutions.pose
    _HAS_MP = True
except Exception:  # keep the module importable without the heavy deps installed
    _HAS_MP = False


# Joints we track, defined as (point_a, vertex, point_c); angle is measured at the vertex.
# Landmark names follow MediaPipe Pose (33 landmarks).
ANGLE_DEFS = {
    "knee_left":   ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "knee_right":  ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "hip_left":    ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "hip_right":   ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "elbow_left":  ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "elbow_right": ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
}


def _angle(a, b, c) -> float:
    """Angle in degrees at vertex b, formed by segments b->a and b->c."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def video_to_angle_series(video_path: str) -> pd.DataFrame:
    """Return a DataFrame: one row per frame, one column per tracked joint angle."""
    if not _HAS_MP:
        raise ImportError("Install mediapipe and opencv-python to run pose extraction.")

    lm_enum = _POSE.PoseLandmark
    rows = []
    with _POSE.Pose(model_complexity=1) as pose:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not res.pose_landmarks:
                continue
            lms = res.pose_landmarks.landmark
            pt = lambda name: np.array([lms[lm_enum[name].value].x,
                                        lms[lm_enum[name].value].y])
            row = {j: _angle(pt(a), pt(b), pt(c)) for j, (a, b, c) in ANGLE_DEFS.items()}
            rows.append(row)
        cap.release()
    return pd.DataFrame(rows)


def summarise(angles: pd.DataFrame) -> dict:
    """Collapse the per-frame angle series into interpretable summary features."""
    feats = {}
    for col in angles.columns:
        s = angles[col].dropna()
        feats[f"{col}_min"] = s.min()
        feats[f"{col}_max"] = s.max()
        feats[f"{col}_rom"] = s.max() - s.min()      # range of motion
        feats[f"{col}_mean"] = s.mean()
        feats[f"{col}_std"] = s.std()                # steadiness
    # left/right symmetry: mean absolute ROM difference
    for side_pair in [("knee_left", "knee_right"),
                      ("hip_left", "hip_right"),
                      ("elbow_left", "elbow_right")]:
        l, r = side_pair
        if l in angles and r in angles:
            feats[f"sym_{l.split('_')[0]}"] = abs(
                (angles[l].max() - angles[l].min()) -
                (angles[r].max() - angles[r].min()))
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    angles = video_to_angle_series(args.video)
    feats = summarise(angles)
    pd.DataFrame([feats]).to_csv(args.out, index=False)
    print(f"Wrote {len(feats)} features to {args.out}")


if __name__ == "__main__":
    main()
