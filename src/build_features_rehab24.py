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
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rehab24_common import iter_reps, parse_exercise_arg, resample_fixed

from paths import rehab24_features

RESAMPLE_LEN = 100
N_PCA_COMPONENTS = 20


def main():
    exercise = parse_exercise_arg()
    out_path = rehab24_features(exercise, "base")

    rows_meta = []
    traj_vecs = []
    for rep in iter_reps(exercise):
        flat = rep.arr.reshape(rep.arr.shape[0], -1)  # (frames, 52)
        traj_vecs.append(resample_fixed(flat, RESAMPLE_LEN).ravel())
        rows_meta.append({"subject": rep.subject, "variant": rep.variant,
                           "rep": rep.rep, "correct": rep.correct})

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

    out = pd.DataFrame(pcs, columns=pd.Index(f"traj_pc{i+1}" for i in range(n_components)))
    out["subject"] = meta["subject"].values
    out["correct"] = meta["correct"].values
    out.to_csv(out_path, index=False)
    print(f"\n{out.shape[0]} ripetizioni, {meta['subject'].nunique()} soggetti, "
          f"{n_components} feature -> {out_path}")
    print(meta.groupby("subject")["correct"].agg(["count", "sum"]))


if __name__ == "__main__":
    main()
