"""
Features for the UI-PRMD deep-squat correct/incorrect data.

Data_Correct.csv / Data_Incorrect.csv (from the Vakanski et al. LSTM-autoencoder
framework repo) hold 90 + 90 repetitions, each already resampled to a fixed 117
frames x 240 dims, and already normalized (not raw joint angles in degrees) -
the exact per-column anatomical meaning is not published alongside this specific
file, so features here treat each repetition's (117, 240) block as a generic
multivariate time series (same trajectory-PCA approach as
build_features_trajectory.py for KIMORE), not hand-picked joint angles.

Two targets are produced: `quality_score` (continuous, from Labels_*.csv) and
`correct` (1 = correct repetition, 0 = incorrect) - the exercise-specific
label KIMORE's cTS could never give us.

Usage:
    python src/build_features_ui_prmd.py
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from paths import RAW, ui_prmd_features

DATA_DIR = RAW / "ui_prmd"
OUT_PATH = ui_prmd_features("deepsquat")

FRAMES_PER_REP = 117
N_PCA_COMPONENTS = 20


def load_reps(data_path: Path, labels_path: Path, correct: int) -> tuple[np.ndarray, pd.DataFrame]:
    data = pd.read_csv(data_path, header=None).values
    labels = pd.read_csv(labels_path, header=None)[0].values
    n_reps = data.shape[0] // FRAMES_PER_REP
    reps = data.reshape(n_reps, FRAMES_PER_REP, data.shape[1])
    meta = pd.DataFrame({"quality_score": labels, "correct": correct})
    return reps, meta


def main():
    reps_c, meta_c = load_reps(DATA_DIR / "Data_Correct.csv", DATA_DIR / "Labels_Correct.csv", 1)
    reps_i, meta_i = load_reps(DATA_DIR / "Data_Incorrect.csv", DATA_DIR / "Labels_Incorrect.csv", 0)

    reps = np.concatenate([reps_c, reps_i], axis=0)
    meta = pd.concat([meta_c, meta_i], axis=0, ignore_index=True)

    flat = reps.reshape(reps.shape[0], -1)  # (n_reps, 117*240)
    scaled = StandardScaler().fit_transform(flat)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    pcs = pca.fit_transform(scaled)

    print("Varianza spiegata per componente:")
    for i, v in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  pc{i}: {v:.3f}")
    print(f"Varianza cumulata con {N_PCA_COMPONENTS} componenti: {pca.explained_variance_ratio_.sum():.3f}")

    out = pd.DataFrame(pcs, columns=pd.Index(f"traj_pc{i+1}" for i in range(N_PCA_COMPONENTS)))
    out["quality_score"] = meta["quality_score"].values
    out["correct"] = meta["correct"].values
    out.to_csv(OUT_PATH, index=False)
    print(f"\n{out.shape[0]} ripetizioni ({meta_c.shape[0]} corrette + {meta_i.shape[0]} scorrette), "
          f"{N_PCA_COMPONENTS} feature -> {OUT_PATH}")


if __name__ == "__main__":
    main()
