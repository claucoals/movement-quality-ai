"""
Dynamics features for REHAB24-6: position-only trajectory PCA (build_features_rehab24.py)
discards how the movement unfolds in time - velocity and acceleration are classic movement-
quality markers in the motor-control literature (jerkier = less controlled), independent of
shape. This extracts position + velocity + acceleration trajectory PCA together, so models
have access to dynamics, not just where the keypoints were.

Velocity/acceleration are finite differences of the (26, 2) keypoint array per frame, computed
before resampling (so they reflect actual frame-to-frame change, not interpolation artifacts),
then each of the three signals (position, velocity, acceleration) is resampled to a fixed
length and PCA'd separately - same safe, no-leakage post-hoc feature construction as the
existing scripts.

Usage:
    python src/build_features_rehab24_dynamics.py --exercise Ex1
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rehab24_common import iter_reps, parse_exercise_arg, resample_fixed

from paths import rehab24_features

RESAMPLE_LEN = 100
N_PCA_COMPONENTS = 15  # per signal (position/velocity/acceleration), so 45 features total


def main():
    exercise = parse_exercise_arg()
    out_path = rehab24_features(exercise, "dynamics")

    rows_meta = []
    traj_pos, traj_vel, traj_acc = [], [], []

    for rep in iter_reps(exercise):
        pos = rep.arr.reshape(rep.arr.shape[0], -1)   # (frames, 52)
        vel = np.diff(pos, axis=0)                     # (frames-1, 52)
        acc = np.diff(vel, axis=0)                      # (frames-2, 52)

        traj_pos.append(resample_fixed(pos, RESAMPLE_LEN).ravel())
        traj_vel.append(resample_fixed(vel, RESAMPLE_LEN).ravel())
        traj_acc.append(resample_fixed(acc, RESAMPLE_LEN).ravel())
        rows_meta.append({"subject": rep.subject, "variant": rep.variant,
                           "rep": rep.rep, "correct": rep.correct})

    meta = pd.DataFrame(rows_meta)
    signal_frames = []
    for name, matrices in [("pos", traj_pos), ("vel", traj_vel), ("acc", traj_acc)]:
        matrix = np.vstack(matrices)
        scaled = StandardScaler().fit_transform(matrix)
        n_components = min(N_PCA_COMPONENTS, scaled.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        pcs = pca.fit_transform(scaled)
        cols = pd.Index(f"{name}_pc{i + 1}" for i in range(n_components))
        signal_frames.append(pd.DataFrame(pcs, columns=cols))
        print(f"  {name}: varianza cumulata {pca.explained_variance_ratio_.sum():.3f} con {n_components} componenti")

    out = pd.concat(signal_frames, axis=1)
    out["subject"] = meta["subject"].values
    out["correct"] = meta["correct"].values
    out.to_csv(out_path, index=False)
    print(f"\n{out.shape[0]} ripetizioni, {meta['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature (posizione+velocita+accelerazione) -> {out_path}")


if __name__ == "__main__":
    main()
