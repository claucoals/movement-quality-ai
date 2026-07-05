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
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from rehab24_common import iter_reps, parse_exercise_arg, resample_fixed

from paths import rehab24_features

RESAMPLE_LEN = 99  # divisible by 3, one point of overlap avoided at phase edges
N_PHASES = 3
N_PCA_PER_PHASE = 8


def main():
    exercise = parse_exercise_arg()
    out_path = rehab24_features(exercise, "phases")

    rows_meta = []
    traj_by_phase = [[] for _ in range(N_PHASES)]
    phase_len = RESAMPLE_LEN // N_PHASES

    for rep in iter_reps(exercise):
        flat = rep.arr.reshape(rep.arr.shape[0], -1)
        resampled = resample_fixed(flat, RESAMPLE_LEN)  # (RESAMPLE_LEN, 52)
        for phase_i in range(N_PHASES):
            chunk = resampled[phase_i * phase_len:(phase_i + 1) * phase_len]
            traj_by_phase[phase_i].append(chunk.ravel())
        rows_meta.append({"subject": rep.subject, "variant": rep.variant,
                           "rep": rep.rep, "correct": rep.correct})

    meta = pd.DataFrame(rows_meta)
    phase_frames = []
    for phase_i in range(N_PHASES):
        matrix = np.vstack(traj_by_phase[phase_i])
        scaled = StandardScaler().fit_transform(matrix)
        n_components = min(N_PCA_PER_PHASE, scaled.shape[0] - 1)
        pca = PCA(n_components=n_components, random_state=42)
        pcs = pca.fit_transform(scaled)
        cols = pd.Index(f"phase{phase_i + 1}_pc{i + 1}" for i in range(n_components))
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
