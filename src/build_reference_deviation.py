"""
Reference trajectories and deviation-from-reference (CoRe-light, "Pipeline B") for REHAB24-6.

SHAP on biophases (Pipeline A) answers "which components does the model rely on". This
answers a different question: "where does a given rep's own movement differ from correct
technique, joint by joint, phase by phase" - the other half needed for the Movement
Attribution Map (SHAP importance x reference deviation).

For each exercise, each of the 11 named joint angles plus the knee-valgus proxy, each of the
3 biomechanical phases (descent/bottom/ascent - same turnaround-based boundaries as
build_features_rehab24_biophases.py, reused here rather than redefined):
  1. slice that phase out of every rep's angle signal
  2. resample it to a fixed length (phases have different frame counts across reps, since the
     turnaround point falls at a different frame each time)
  3. average the resampled phase-trajectories of *correct* reps from every OTHER subject ->
     that subject's reference (leave-subject-out, same anti-leakage rule enforced everywhere
     else in this project via quality_model.py's grouped CV - a rep's deviation must not be
     measured against a reference partly built from its own subject's other reps, or any
     downstream check of "does deviation predict correct" would be inflated by subjects simply
     being closer to their own average than to other people's)
  4. every rep's deviation = mean absolute difference between its own resampled
     phase-trajectory and its subject's leave-subject-out reference

This remains usable both as a descriptive/interpretability artifact and, because the
leave-subject-out reference makes it leakage-safe, as a candidate model feature.

Usage:
    python src/build_reference_deviation.py --exercise Ex1
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from build_features_rehab24_anatomical import ANGLE_DEFS, angle_series, knee_valgus_proxy
from build_features_rehab24_biophases import PHASES, find_phase_bounds, reference_angle_name
from rehab24_common import iter_reps, parse_exercise_arg

from paths import rehab24_features

RESAMPLE_N = 20  # points per phase after resampling - a fixed length is what makes averaging
                 # and differencing trajectories across reps of different durations possible


def resample(signal: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    if len(signal) == 1:
        return np.full(n, signal[0])
    t_old = np.linspace(0.0, 1.0, len(signal))
    t_new = np.linspace(0.0, 1.0, n)
    return np.interp(t_new, t_old, signal)


def phase_trajectories(arr: np.ndarray, exercise: str, variant: str) -> dict:
    """Returns {(phase, angle_name): resampled trajectory} for one rep, angle_name ranging
    over every named joint angle plus 'knee_valgus'."""
    ref_signal = angle_series(arr, *ANGLE_DEFS[reference_angle_name(exercise, variant)])
    lo, hi = find_phase_bounds(ref_signal)
    bounds = {"descent": (0, lo), "bottom": (lo, hi), "ascent": (hi, len(arr))}

    out = {}
    for phase in PHASES:
        a, b = bounds[phase]
        sub_arr = arr[a:b]
        for angle_name, (ja, jb, jc) in ANGLE_DEFS.items():
            out[(phase, angle_name)] = resample(angle_series(sub_arr, ja, jb, jc))
        out[(phase, "knee_valgus")] = resample(knee_valgus_proxy(sub_arr))
    return out


def build_reference(correct_reps: list[dict], angle_names: list[str]) -> dict:
    reference = {}
    for phase in PHASES:
        for angle_name in angle_names:
            key = (phase, angle_name)
            stacked = np.stack([r["traj"][key] for r in correct_reps])
            reference[key] = stacked.mean(axis=0)
    return reference


def main():
    exercise = parse_exercise_arg()

    reps = [{"subject": rep.subject, "variant": rep.variant, "rep": rep.rep,
             "correct": rep.correct, "arr": rep.arr} for rep in iter_reps(exercise)]

    for rep in reps:
        rep["traj"] = phase_trajectories(rep["arr"], exercise, rep["variant"])

    correct_reps = [r for r in reps if r["correct"] == 1]
    print(f"{len(correct_reps)} ripetizioni corrette usate come riferimento "
          f"(su {len(reps)} totali)")

    angle_names = list(ANGLE_DEFS.keys()) + ["knee_valgus"]
    subjects = sorted({r["subject"] for r in reps})
    # one reference per held-out subject, built only from OTHER subjects' correct reps -
    # otherwise a subject's own reps would help define the very reference it's compared
    # against, understating its deviation relative to a genuinely unseen reference
    reference_by_subject = {
        held_out: build_reference([r for r in correct_reps if r["subject"] != held_out], angle_names)
        for held_out in subjects
    }

    rows = []
    for rep in reps:
        reference = reference_by_subject[rep["subject"]]
        row = {"subject": rep["subject"], "correct": rep["correct"]}
        for phase in PHASES:
            for angle_name in angle_names:
                key = (phase, angle_name)
                row[f"deviation__{phase}__{angle_name}"] = np.abs(rep["traj"][key] - reference[key]).mean()
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = rehab24_features(exercise, "deviation")
    out.to_csv(out_path, index=False)
    print(f"{out.shape[0]} ripetizioni, {out['subject'].nunique()} soggetti, "
          f"{out.shape[1] - 2} feature di deviazione ({len(PHASES)} fasi x {len(angle_names)} "
          f"articolazioni) -> {out_path}")


if __name__ == "__main__":
    main()
