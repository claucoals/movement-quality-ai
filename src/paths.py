"""Canonical paths for data/ and results/. Import from build and run scripts."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
FEATURES = DATA / "features"
RESULTS = ROOT / "results"
EXPERIMENTS_CSV = RESULTS / "experiments" / "experiments.csv"
SHAP_DIR = RESULTS / "shap"


def kimore_features(stem: str) -> Path:
    return FEATURES / "kimore" / f"{stem}.csv"


def ui_prmd_features(stem: str) -> Path:
    return FEATURES / "ui_prmd" / f"{stem}.csv"


def rehab24_features(exercise: str, family: str) -> Path:
    """family: base | dynamics | anatomical | biophases | phases | deviation"""
    return FEATURES / "rehab24" / exercise.lower() / f"{family}.csv"


def rehab24_pooled_anatomical() -> Path:
    return FEATURES / "rehab24" / "pooled_anatomical.csv"
