"""
Features for the REHAB24-6 dataset (2D pose keypoints, segmented per repetition).

Each .npy file is one repetition: (n_frames, 26, 2) array of 2D joint pixel
coordinates from pose estimation, already segmented. The filename encodes an
exercise variant, repetition number, and label (1 = correct, 0 = incorrect):
e.g. PM_001_c18_rightarm-1-rep6-0.npy -> rep 6, label 0. Subject id is NOT
taken from the filename's "PM_NNN" string - see rehab24_annotations.py for
why (two different filename strings can be the same real person) - it comes
from data/raw/rehab24/annotations.csv's person_id via subject_id_for().

Reps flagged `mocap_erroneous` in that same file are excluded.

Exact keypoint identities (which of the 26 points is which joint) are not
given alongside this file, so - as with UI-PRMD - features are a generic
trajectory-PCA representation (resample each rep to a fixed number of frames,
flatten, PCA across repetitions), not hand-picked joint angles.

Usage:
    python src/build_features_rehab24.py --exercise Ex1
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rehab24_annotations import is_mocap_erroneous, subject_id_for

ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "data" / "raw" / "rehab24"

FILENAME_RE = re.compile(r"^(PM_\w+)_c(\d+)_(.+)-rep(\d+)-(\d+)\.npy$")
RESAMPLE_LEN = 100
N_PCA_COMPONENTS = 20


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
    out_path = ROOT / "data" / f"features_rehab24_{args.exercise.lower()}.csv"

    rows_meta = []
    traj_vecs = []
    n_skipped, n_mocap_erroneous = 0, 0
    for f in sorted(ex_dir.iterdir()):
        if f.suffix != ".npy":
            continue
        m = FILENAME_RE.match(f.name)
        if not m:
            n_skipped += 1
            continue
        if is_mocap_erroneous(f.name):
            n_mocap_erroneous += 1
            continue
        _, cam, variant, rep, label = m.groups()
        arr = np.load(f)  # (frames, 26, 2)
        flat = arr.reshape(arr.shape[0], -1)  # (frames, 52)
        traj_vecs.append(resample_fixed(flat).ravel())
        rows_meta.append({"subject": subject_id_for(f.name), "variant": variant,
                           "rep": int(rep), "correct": int(label)})

    print(f"{len(rows_meta)} ripetizioni caricate, {n_skipped} file scartati (nome non conforme), "
          f"{n_mocap_erroneous} scartati (mocap_erroneous)")

    meta = pd.DataFrame(rows_meta)
    traj_matrix = np.vstack(traj_vecs)

    scaled = StandardScaler().fit_transform(traj_matrix)
    n_components = min(N_PCA_COMPONENTS, scaled.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(scaled)

    print("Varianza spiegata per componente:")
    for i, v in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  pc{i}: {v:.3f}")
    print(f"Varianza cumulata con {n_components} componenti: {pca.explained_variance_ratio_.sum():.3f}")

    out = pd.DataFrame(pcs, columns=[f"traj_pc{i+1}" for i in range(n_components)])
    out["subject"] = meta["subject"].values
    out["correct"] = meta["correct"].values
    out.to_csv(out_path, index=False)
    print(f"\n{out.shape[0]} ripetizioni, {meta['subject'].nunique()} soggetti, "
          f"{n_components} feature -> {out_path}")
    print(meta.groupby("subject")["correct"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()
