"""
Structurally different feature family for KIMORE ex5 (squat), one row per
subject: not ROM/velocity/tempo summary stats anymore, but (a) shape of the
whole trajectory via PCA on a fixed-length resample of the angle signals, and
(b) frequency-domain descriptors of how periodic/regular the movement is.

Rationale: the previous feature family (build_features_ex5.py) showed zero
univariate signal against cTS, in ex5 and in every other KIMORE exercise. This
tries a structurally different representation of the same underlying
kinematic signal before concluding the target itself is unlearnable from
Kinect skeleton data at this sample size.

Usage:
    python src/build_features_trajectory.py
"""

from __future__ import annotations
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PKL_PATH = ROOT / "data" / "kimore_exercise_dataset.pkl"
OUT_PATH = ROOT / "data" / "features_trajectory_ex5.csv"

ANGLE_DEFS = {
    "knee_left":  ("hipleft", "kneeleft", "ankleleft"),
    "knee_right": ("hipright", "kneeright", "ankleright"),
    "hip_left":   ("shoulderleft", "hipleft", "kneeleft"),
    "hip_right":  ("shoulderright", "hipright", "kneeright"),
}
RESAMPLE_LEN = 100
N_PCA_COMPONENTS = 10


def positions(row: pd.Series, joint: str) -> np.ndarray:
    return np.asarray(row[joint])[:, 4:7]


def angle_series(row: pd.Series, a: str, b: str, c: str) -> np.ndarray:
    pa, pb, pc = positions(row, a), positions(row, b), positions(row, c)
    ba, bc = pa - pb, pc - pb
    cos = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def resample_fixed(signal: np.ndarray, n: int = RESAMPLE_LEN) -> np.ndarray:
    """Whole recording mapped to a fixed number of points via linear
    interpolation over normalized time, so subjects with different recording
    lengths / rep counts become directly comparable in shape."""
    t_old = np.linspace(0.0, 1.0, len(signal))
    t_new = np.linspace(0.0, 1.0, n)
    return np.interp(t_new, t_old, signal)


def frequency_features(signal: np.ndarray) -> dict:
    """Spectral descriptors of how periodic/regular the movement is, not tied
    to absolute frame rate: dominant frequency position, how peaky the
    spectrum is, and its Shannon entropy (all normalized, so comparable across
    recordings of different length)."""
    detrended = signal - signal.mean()
    power = np.abs(np.fft.rfft(detrended)) ** 2
    power = power[1:]  # drop the DC bin
    total = power.sum()
    if total < 1e-9 or len(power) == 0:
        return {"dominant_freq_norm": np.nan, "spectral_peak_ratio": np.nan, "spectral_entropy": np.nan}
    p_norm = power / total
    dominant_freq_norm = np.argmax(power) / len(power)
    spectral_peak_ratio = p_norm.max()
    spectral_entropy = -np.sum(p_norm * np.log(p_norm + 1e-12)) / np.log(len(p_norm))
    return {
        "dominant_freq_norm": dominant_freq_norm,
        "spectral_peak_ratio": spectral_peak_ratio,
        "spectral_entropy": spectral_entropy,
    }


def main():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    ex5 = data["ex5"]

    freq_rows = []
    traj_rows = []
    for _, row in ex5.iterrows():
        freq_feats = {}
        traj_vec = []
        for name, (a, b, c) in ANGLE_DEFS.items():
            s = angle_series(row, a, b, c)
            for k, v in frequency_features(s).items():
                freq_feats[f"{name}_{k}"] = v
            traj_vec.append(resample_fixed(s))
        freq_rows.append(freq_feats)
        traj_rows.append(np.concatenate(traj_vec))

    freq_df = pd.DataFrame(freq_rows)
    traj_matrix = np.vstack(traj_rows)  # (n_subjects, 4 * RESAMPLE_LEN)

    scaled = StandardScaler().fit_transform(traj_matrix)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    pcs = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(pcs, columns=[f"traj_pc{i+1}" for i in range(N_PCA_COMPONENTS)])

    print("Varianza spiegata per componente:")
    for i, v in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  pc{i}: {v:.3f}")
    print(f"Varianza cumulata con {N_PCA_COMPONENTS} componenti: {pca.explained_variance_ratio_.sum():.3f}")

    out = pd.concat([freq_df, pca_df], axis=1)
    out["cTS"] = ex5["cTS"].values
    out.to_csv(OUT_PATH, index=False)
    print(f"\n{out.shape[0]} soggetti, {out.shape[1] - 1} feature + target -> {OUT_PATH}")


if __name__ == "__main__":
    main()
