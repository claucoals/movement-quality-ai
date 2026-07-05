"""
Shared loading/filtering logic for every REHAB24-6 feature-builder script (build_features_
rehab24*.py). Each of those scripts used to repeat this same block: parse the rep filename,
skip malformed names, exclude mocap_erroneous reps, resolve subject id via annotations.csv
(not the filename's "PM_NNN" string - see rehab24_annotations.py), and print the same
loaded/skipped/excluded counts. Only the feature computation itself differs between scripts.
"""

from __future__ import annotations
import argparse
import re
from typing import Iterator, NamedTuple

import numpy as np

from paths import RAW
from rehab24_annotations import is_mocap_erroneous, subject_id_for

BASE_DIR = RAW / "rehab24"
FILENAME_RE = re.compile(r"^(PM_\w+)_c(\d+)_(.+)-rep(\d+)-(\d+)\.npy$")


class Rep(NamedTuple):
    arr: np.ndarray  # (n_frames, 26, 2) raw keypoints
    subject: str
    variant: str
    rep: int
    correct: int


def parse_exercise_arg() -> str:
    p = argparse.ArgumentParser()
    p.add_argument("--exercise", required=True, help="e.g. Ex1")
    return p.parse_args().exercise


def iter_reps(exercise: str) -> Iterator[Rep]:
    """Yields one Rep per valid, non-mocap-erroneous file in <exercise>-segmented/, then
    prints the loaded/skipped/excluded counts every feature-builder script already reported."""
    ex_dir = BASE_DIR / f"{exercise}-segmented"
    n_loaded, n_skipped, n_mocap_erroneous = 0, 0, 0
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
        _, _cam, variant, rep, label = m.groups()
        n_loaded += 1
        yield Rep(np.load(f), subject_id_for(f.name), variant, int(rep), int(label))
    print(f"{n_loaded} ripetizioni caricate, {n_skipped} file scartati (nome non conforme), "
          f"{n_mocap_erroneous} scartati (mocap_erroneous)")


def resample_fixed(signal: np.ndarray, n: int) -> np.ndarray:
    t_old = np.linspace(0.0, 1.0, len(signal))
    t_new = np.linspace(0.0, 1.0, n)
    out = np.empty((n, signal.shape[1]))
    for j in range(signal.shape[1]):
        out[:, j] = np.interp(t_new, t_old, signal[:, j])
    return out
