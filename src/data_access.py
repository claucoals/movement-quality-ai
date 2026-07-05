"""
Dataset access — instructions only. No raw data / videos are committed.

KIMORE (KInematic Assessment of MOvement, low-back rehab, expert quality scores):
    Request/download via the authors' page (Capecci et al., 2019, IEEE TNSRE).
    Search "KIMORE dataset Univ. Politecnica delle Marche". Place under data/raw/kimore/

UI-PRMD (University of Idaho Physical Rehabilitation Movement Data):
    https://webpages.uidaho.edu/ui-prmd/  -> data/raw/ui_prmd/
    Ten exercises, correct and incorrect repetitions, Kinect + Vicon skeletons.

REHAB24-6:
    Zenodo record 13305826. 6 exercises, 2D pose keypoints, subject id + correct/incorrect
    per rep, plus per-rep QC metadata (annotations.csv - see rehab24_annotations.py for why
    that file matters beyond being "extra" data). -> data/raw/rehab24/

IntelliRehabDS (IRDS):
    Public via MDPI Data (2021). Correctness labels. -> data/raw/irds/

Self-recorded Pilates:
    Record a few exercises (e.g. Hundred, Roll-Up, Shoulder Bridge, Single-Leg Stretch),
    each performed correctly and with 1-2 common faults, several reps, a few people.
    Get simple written consent. Place clips under data/raw/pilates/ (git-ignored).

Directory layout (derived artifacts under data/features/, not committed):
    data/raw/{kimore,ui_prmd,rehab24}/     raw downloads
    data/features/kimore/                    ex5.csv, ex5_classification.csv
    data/features/ui_prmd/                 classification.csv, regression.csv, deepsquat.csv
    data/features/rehab24/ex{N}/            base, dynamics, anatomical, biophases, phases, deviation
    data/features/rehab24/pooled_anatomical.csv   leave-one-exercise-out input
    results/experiments/experiments.csv    nested-CV sweep output (run_experiments.py)
    results/shap/                          SHAP CSVs (run_shap.py)

Canonical path helpers live in src/paths.py.
"""

from paths import RAW


def check_data():
    for name in ("kimore", "ui_prmd", "rehab24", "irds", "pilates"):
        d = RAW / name
        print(f"{name:8s}: {d}  [{'found' if d.exists() else 'MISSING'}]")


if __name__ == "__main__":
    check_data()
