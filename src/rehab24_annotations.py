"""
Shared lookup for data/raw/rehab24/annotations.csv - ground-truth per-repetition metadata
released with REHAB24-6 that every feature-building script had been ignoring (it sat unused
inside a nested raw-data folder). Three facts from it matter for correctness, not just extra
information:

1. `person_id` is the TRUE subject identity - the filename's "PM_NNN" string is not a reliable
   stand-in for it. Checked across all 6 exercises: Ex1 (PM_000 and PM_001), Ex2 (PM_013 and
   PM_014) and Ex5 (PM_117a and PM_117b) each have two different filename subject-strings that
   are actually the same person_id. Every feature-building script was using the filename string
   as the CV `groups` column, which means the anti-leakage guarantee this project is built
   around (no subject in both train and test) was silently violated for those three exercises -
   reps from the same real person could land on both sides of a split. subject_id_for() below
   is the fix: always derive the CV group from person_id, never from the filename.

2. `mocap_erroneous`: 11 of 606 reps (all from one subject, Ex3/push-ups) are flagged by the
   dataset creators as having bad mocap tracking. These should be excluded from every feature
   family, the same way a corrupted sample would be in any other pipeline - not a judgment
   call, the dataset itself says so.

3. `exercise_subtype`: Ex1 is 100% "rightarm", Ex4 is a mix of "leftleg"/"rightleg", Ex5 is a
   mix of "frontlegleft"/"frontlegright" (already present in the filename's variant field,
   parsed by every script's FILENAME_RE but previously discarded). Any feature that commits to
   one side (e.g. the phase-turnaround detection in build_features_rehab24_biophases.py, which
   picks a single reference angle) was silently reading the wrong limb for every "rightleg" /
   "rightarm" / "frontlegright" rep. Ex2/Ex3/Ex6 have no variant (bilateral movements), so a
   fixed side is fine there.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
_ANNOTATIONS = pd.read_csv(ROOT / "data" / "raw" / "rehab24" / "annotations.csv").set_index("file_name")


def is_mocap_erroneous(filename: str) -> bool:
    if filename not in _ANNOTATIONS.index:
        return False
    return bool(_ANNOTATIONS.loc[filename, "mocap_erroneous"])


def subject_id_for(filename: str) -> str:
    """True subject id (annotations.csv's person_id, not the filename's "PM_NNN" string -
    see module docstring for why that distinction matters). person_id is only unique within
    one exercise's own recording session, never combined across exercises by subject, so that
    is not a limitation here."""
    return f"P{int(_ANNOTATIONS.loc[filename, 'person_id']):03d}"


def side_for(exercise: str, variant: str) -> str:
    """Which side ('l' or 'r') this rep's variant targets, for exercises with a left/right
    split. Exercises without one (no "left"/"right" in the variant string) default to 'l' -
    an arbitrary but harmless choice since both sides move together."""
    v = variant.lower()
    if "right" in v:
        return "r"
    return "l"
