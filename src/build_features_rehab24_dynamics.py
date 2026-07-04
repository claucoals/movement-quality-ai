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
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "ui_prmd" / "2d_joints_segmented" / "2d_joints_segmented"

FILENAME_RE = re.compile(r"^(PM_\d+)_c(\d+)_(.+)-rep(\d+)-(\d+)\.npy$")
RESAMPLE_LEN = 100
N_PCA_COMPONENTS = 15  # per signal (position/velocity/acceleration), so 45 features total


def resample_fixed(signal: np.ndarray, n: int = RESAMPLE_LEN) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, len(signal))
    t_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, signal.shape[1]))
    for j in range(signal.shape[1]):
        out[:, j] = np.interp(t_new, t_old, signal[:, j])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exercise", required=True, help="e.g. Ex1")
    args = p.parse_args()

    ex_dir = BASE_DIR / f"{args.exercise}-segmented"
    out_path = ROOT / "data" / f"features_rehab24_{args.exercise.lower()}_dynamics.csv"

    rows_meta = []
    traj_pos, traj_vel, traj_acc = [], [], []
    n_skipped = 0

    for f in sorted(ex_dir.iterdir()):
        if f.suffix != ".npy":
            continue
        m = FILENAME_RE.match(f.name)
        if not m:
            n_skipped += 1
            continue
        subject, cam, variant, rep, label = m.groups()
        arr = np.load(f)  # (frames, 26, 2)
        pos = arr.reshape(arr.shape[0], -1)          # (frames, 52)
        vel = np.diff(pos, axis=0)                    # (frames-1, 52)
        acc = np.diff(vel, axis=0)                    # (frames-2, 52)

        traj_pos.append(resample_fixed(pos).ravel())
        traj_vel.append(resample_fixed(vel).ravel())
        traj_acc.append(resample_fixed(acc).ravel())
        rows_meta.append({"subject": subject, "variant": variant, "rep": int(rep), "correct": int(label)})

    print(f"{len(rows_meta)} ripetizioni caricate, {n_skipped} file scartati (nome non conforme)")

    meta = pd.DataFrame(rows_meta)
    signal_frames = []
    for name, matrices in [("pos", traj_pos), ("vel", traj_vel), ("acc", traj_acc)]:
        matrix = np.vstack(matrices)
        scaled = StandardScaler().fit_transform(matrix)
        n_components = min(N_PCA_COMPONENTS, scaled.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        pcs = pca.fit_transform(scaled)
        cols = [f"{name}_pc{i + 1}" for i in range(n_components)]
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
