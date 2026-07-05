"""
Phase-segmented features for REHAB24-6: same trajectory-PCA idea as
build_features_rehab24.py, but each repetition is split into 3 equal temporal
phases (early / mid / late third) before PCA, each phase reduced separately.

Rationale (Bai et al. 2022 "Temporal Parsing Transformer", Xu et al. 2024
"FineParser"): action quality often depends on WHEN in the movement the error
happens, not just the movement as a whole. This is the same idea kept light
and interpretable (rule-based equal-thirds split, not a learned parser):
does knowing which third of the rep the deviation falls in help, or is one
phase driving the signal more than the others?

Usage:
    python src/build_features_rehab24_phases.py --exercise Ex1
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
RESAMPLE_LEN = 99  # divisible by 3, one point of overlap avoided at phase edges
N_PHASES = 3
N_PCA_PER_PHASE = 8


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
    out_path = ROOT / "data" / f"features_rehab24_{args.exercise.lower()}_phases.csv"

    rows_meta = []
    traj_by_phase = [[] for _ in range(N_PHASES)]
    n_skipped, n_mocap_erroneous = 0, 0
    phase_len = RESAMPLE_LEN // N_PHASES

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
        arr = np.load(f)
        flat = arr.reshape(arr.shape[0], -1)
        resampled = resample_fixed(flat)  # (RESAMPLE_LEN, 52)
        for phase_i in range(N_PHASES):
            chunk = resampled[phase_i * phase_len:(phase_i + 1) * phase_len]
            traj_by_phase[phase_i].append(chunk.ravel())
        rows_meta.append({"subject": subject_id_for(f.name), "variant": variant,
                           "rep": int(rep), "correct": int(label)})

    print(f"{len(rows_meta)} ripetizioni caricate, {n_skipped} file scartati (nome non conforme), "
          f"{n_mocap_erroneous} scartati (mocap_erroneous)")

    meta = pd.DataFrame(rows_meta)
    phase_frames = []
    for phase_i in range(N_PHASES):
        matrix = np.vstack(traj_by_phase[phase_i])
        scaled = StandardScaler().fit_transform(matrix)
        n_components = min(N_PCA_PER_PHASE, scaled.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        pcs = pca.fit_transform(scaled)
        cols = [f"phase{phase_i + 1}_pc{i + 1}" for i in range(n_components)]
        phase_frames.append(pd.DataFrame(pcs, columns=cols))
        print(f"  fase {phase_i + 1}: varianza cumulata {pca.explained_variance_ratio_.sum():.3f} "
              f"con {n_components} componenti")

    out = pd.concat(phase_frames, axis=1)
    out["subject"] = meta["subject"].values
    out["correct"] = meta["correct"].values
    out.to_csv(out_path, index=False)
    print(f"\n{out.shape[0]} ripetizioni, {meta['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature ({N_PHASES} fasi x {N_PCA_PER_PHASE} pc) -> {out_path}")


if __name__ == "__main__":
    main()
