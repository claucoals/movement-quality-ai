"""
video -> pose landmarks -> the same anatomical joint-angle feature schema REHAB24 uses.

Front-end is MediaPipe Pose (fast, runs on a laptop, no GPU needed). Column names and summary
statistics come from anatomical_features.py, the same module build_features_rehab24_anatomical.py
uses - deliberate: a Pilates clip run through this script produces the identical feature vector
schema REHAB24's anatomical models are trained on, so a model trained on REHAB24 can be tested
on Pilates footage instead of being blocked by mismatched columns.

Two angles need a landmark MediaPipe's 33-point model doesn't have directly:
  - l/r_shoulder_angle needs REHAB24's "neck" - approximated below as the shoulder midpoint.
  - trunk_flex needs REHAB24's "spine1" as the vertex of a 3-point angle (neck-spine1-hips).
    MediaPipe has no intermediate spine landmark, and a synthetic midpoint vertex would be
    degenerate (neck and hips would sit ~equidistant on either side of their own midpoint,
    giving ~180 degrees almost regardless of real trunk lean). Computed instead as the
    neck-to-hip line's angle from vertical - a standard way to measure trunk flexion, but NOT
    the same geometry as REHAB24's vertex angle. Flagged here rather than silently treated as
    equivalent: expect this one column to correspond less precisely across the two sources
    than the others.

knee_valgus carries a coronal-plane risk: monocular RGB pose estimation is specifically less
reliable for this angle than for sagittal-plane ones (systematic error/variability reported in
the markerless-mocap validity literature) - the formula is identical to REHAB24's, the two
inputs are not equally trustworthy. Frontal camera placement matters more for this feature
than any other here.

Usage:
    python pose_to_features.py --video path/to/clip.mp4 --out features.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from anatomical_features import ANGLE_NAMES, angle_at_vertex, knee_valgus_ratio, summary_features

try:
    import cv2
    import mediapipe as mp
    _POSE = mp.solutions.pose  # pyright: ignore[reportAttributeAccessIssue]
    _HAS_MP = True
except Exception:  # keep the module importable without the heavy deps installed
    cv2 = None
    _HAS_MP = False


# (point_a, vertex, point_c) triples in MediaPipe landmark names, keyed by REHAB24's own
# anatomical angle names (anatomical_features.ANGLE_NAMES) so both sources produce identical
# columns. "neck" is synthetic (shoulder midpoint, added in _stack_landmarks), not a real
# MediaPipe landmark. trunk_flex is handled separately - see module docstring.
ANGLE_DEFS = {
    "l_elbow":          ("LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST"),
    "r_elbow":          ("RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"),
    "l_shoulder_angle": ("neck", "LEFT_SHOULDER", "LEFT_ELBOW"),
    "r_shoulder_angle": ("neck", "RIGHT_SHOULDER", "RIGHT_ELBOW"),
    "l_knee":           ("LEFT_HIP", "LEFT_KNEE", "LEFT_ANKLE"),
    "r_knee":           ("RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"),
    "l_hip":            ("LEFT_SHOULDER", "LEFT_HIP", "LEFT_KNEE"),
    "r_hip":            ("RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"),
    "l_ankle":          ("LEFT_KNEE", "LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    "r_ankle":          ("RIGHT_KNEE", "RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
}
assert set(ANGLE_DEFS) | {"trunk_flex"} == set(ANGLE_NAMES), \
    "ANGLE_DEFS + trunk_flex must cover exactly the shared anatomical_features.ANGLE_NAMES schema"

_REQUIRED_LANDMARKS = sorted({name for triple in ANGLE_DEFS.values() for name in triple
                               if name != "neck"})


def _stack_landmarks(video_path: str) -> dict[str, np.ndarray]:
    """One (n_frames, 2) array per required landmark (plus synthetic 'neck'), skipping frames
    where MediaPipe found no pose - same behaviour as the pre-existing code, just collected
    per-landmark instead of per-angle so knee_valgus_ratio's per-trial median floor (see
    anatomical_features.py) sees the whole trial at once, not one frame at a time."""
    if not _HAS_MP:
        raise ImportError("Install mediapipe and opencv-python to run pose extraction.")
    assert cv2 is not None  # narrows for the type checker; guaranteed by _HAS_MP above

    lm_enum = _POSE.PoseLandmark
    frames = []
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
            frames.append({name: np.array([lms[lm_enum[name].value].x, lms[lm_enum[name].value].y])
                            for name in _REQUIRED_LANDMARKS})
        cap.release()

    points = {name: np.stack([f[name] for f in frames]) for name in _REQUIRED_LANDMARKS}
    points["neck"] = (points["LEFT_SHOULDER"] + points["RIGHT_SHOULDER"]) / 2
    return points


def _trunk_flex_series(neck: np.ndarray, hips_mid: np.ndarray) -> np.ndarray:
    """Angle of the neck->hip-midpoint line from vertical - see module docstring for why this
    isn't the same 3-point vertex angle REHAB24 uses for trunk_flex."""
    trunk_vec = neck - hips_mid
    vertical = np.array([0.0, -1.0])  # image y grows downward, "up" is -y
    cos = (np.sum(trunk_vec * vertical, axis=-1) /
           (np.linalg.norm(trunk_vec, axis=-1) * np.linalg.norm(vertical) + 1e-9))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def video_to_angle_series(video_path: str) -> pd.DataFrame:
    """Return a DataFrame: one row per frame, one column per anatomical_features.ANGLE_NAMES
    angle, plus knee_valgus."""
    points = _stack_landmarks(video_path)
    angles = {name: angle_at_vertex(points[a], points[b], points[c])
              for name, (a, b, c) in ANGLE_DEFS.items()}
    hips_mid = (points["LEFT_HIP"] + points["RIGHT_HIP"]) / 2
    angles["trunk_flex"] = _trunk_flex_series(points["neck"], hips_mid)

    df = pd.DataFrame(angles)
    df["knee_valgus"] = knee_valgus_ratio(points["LEFT_KNEE"], points["RIGHT_KNEE"],
                                           points["LEFT_HIP"], points["RIGHT_HIP"])
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    angles = video_to_angle_series(args.video)
    angle_signals = {name: angles[name].dropna().to_numpy() for name in ANGLE_NAMES}
    feats = summary_features(angle_signals, angles["knee_valgus"].dropna().to_numpy())
    pd.DataFrame([feats]).to_csv(args.out, index=False)
    print(f"Wrote {len(feats)} features to {args.out}")


if __name__ == "__main__":
    main()
