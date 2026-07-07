"""
Shared angle-signal -> summary-feature logic for anatomical (named joint-angle) features.

Used by both build_features_rehab24_anatomical.py (mocap-derived REHAB24 keypoints) and
pose_to_features.py (MediaPipe smartphone video) - kept in one place specifically so a
Pilates clip and a REHAB24 rep land in the identical feature-vector schema: same column
names, same summary statistics, same knee-valgus definition. Without this, "train on
REHAB24, test on Pilates" is architecturally impossible, not just underpowered, because the
feature spaces wouldn't match.

Each caller supplies its own joint-position lookup (REHAB24's 26-joint mocap indexing vs
MediaPipe's 33 landmarks are different schemas) - this module only knows about raw point
arrays and per-frame angle series, never where they came from.
"""

from __future__ import annotations
import numpy as np

# The 11 named angles every anatomical feature table (REHAB24 or Pilates/MediaPipe) shares.
ANGLE_NAMES = ["l_elbow", "r_elbow", "l_shoulder_angle", "r_shoulder_angle",
               "l_knee", "r_knee", "l_hip", "r_hip", "l_ankle", "r_ankle", "trunk_flex"]

_SYMMETRY_PAIRS = [("elbow", "l_elbow", "r_elbow"), ("knee", "l_knee", "r_knee"),
                   ("hip", "l_hip", "r_hip"), ("shoulder", "l_shoulder_angle", "r_shoulder_angle")]


def angle_at_vertex(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """Angle in degrees at vertex b, formed by rays b->a and b->c. pa/pb/pc are (n_frames, 2)
    (or broadcastable) point arrays."""
    ba, bc = pa - pb, pc - pb
    cos = (np.sum(ba * bc, axis=-1) /
           (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-9))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def knee_valgus_ratio(l_knee: np.ndarray, r_knee: np.ndarray,
                       l_hip: np.ndarray, r_hip: np.ndarray) -> np.ndarray:
    """Ratio of inter-knee to inter-hip lateral (x) distance: well below 1 means the knees are
    caving in relative to the hips (valgus), a classic squat/lunge quality fault. hip_dist is
    floored at a fraction of its own trial median rather than a fixed epsilon: it can
    genuinely collapse toward zero from ordinary 2D-projection foreshortening (not a tracking
    error) when the hips momentarily align along the camera's viewing axis, and a fixed
    epsilon does nothing against a real small denominator. Known limitation: monocular RGB
    pose estimation is specifically less reliable for this, coronal-plane, angle than for
    sagittal-plane ones - this formula is the same on REHAB24 and on phone video, but the two
    inputs are not equally trustworthy."""
    knee_dist = np.abs(l_knee[..., 0] - r_knee[..., 0])
    hip_dist = np.abs(l_hip[..., 0] - r_hip[..., 0])
    floor = 0.25 * np.median(hip_dist)
    return knee_dist / np.maximum(hip_dist, floor)


def summary_features(angle_series: dict[str, np.ndarray], valgus_series: np.ndarray) -> dict:
    """angle_series: {name in ANGLE_NAMES: 1D array over frames}. Returns
    min/max/rom/mean/std/vel_mean_abs/vel_max_abs per angle, L/R symmetry (abs ROM
    difference), and knee-valgus min/mean - the same feature set
    build_features_rehab24_anatomical.subject_features computes on REHAB24, column-for-column,
    so a model trained on one input can score the other."""
    feats = {}
    for name, s in angle_series.items():
        v = np.diff(s)
        feats[f"{name}_min"] = s.min()
        feats[f"{name}_max"] = s.max()
        feats[f"{name}_rom"] = s.max() - s.min()
        feats[f"{name}_mean"] = s.mean()
        feats[f"{name}_std"] = s.std()
        feats[f"{name}_vel_mean_abs"] = np.abs(v).mean() if len(v) else 0.0
        feats[f"{name}_vel_max_abs"] = np.abs(v).max() if len(v) else 0.0

    for label, l_name, r_name in _SYMMETRY_PAIRS:
        if l_name in angle_series and r_name in angle_series:
            feats[f"sym_{label}"] = abs(feats[f"{l_name}_rom"] - feats[f"{r_name}_rom"])

    feats["knee_valgus_min"] = valgus_series.min()
    feats["knee_valgus_mean"] = valgus_series.mean()
    return feats
