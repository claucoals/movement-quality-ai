"""
Dataset access — instructions only. No raw data / videos are committed.

KIMORE (KInematic Assessment of MOvement, low-back rehab, expert quality scores):
    Request/download via the authors' page (Capecci et al., 2019, IEEE TNSRE).
    Search "KIMORE dataset Univ. Politecnica delle Marche". Place under data/raw/kimore/

UI-PRMD (University of Idaho Physical Rehabilitation Movement Data):
    https://webpages.uidaho.edu/ui-prmd/  -> data/raw/ui_prmd/
    Ten exercises, correct and incorrect repetitions, Kinect + Vicon skeletons.

IntelliRehabDS (IRDS):
    Public via MDPI Data (2021). Correctness labels. -> data/raw/irds/

Self-recorded Pilates (Layer 2):
    Record a few exercises (e.g. Hundred, Roll-Up, Shoulder Bridge, Single-Leg Stretch),
    each performed correctly and with 1-2 common faults, several reps, a few people.
    Get simple written consent. Place clips under data/raw/pilates/ (git-ignored).
"""

from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def check_data():
    for name in ("kimore", "ui_prmd", "irds", "pilates"):
        d = RAW / name
        print(f"{name:8s}: {d}  [{'found' if d.exists() else 'MISSING'}]")


if __name__ == "__main__":
    check_data()
