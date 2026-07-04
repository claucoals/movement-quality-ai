# Model Card

## How to read this card
Every number here is reproducible: `python src/run_experiments.py` regenerates
`results/experiments.csv` from `config.yaml`, and `notebooks/05_experiments_analysis.ipynb`
turns that CSV into the tables/plots this card summarizes. Nothing below is hand-typed from a
one-off run - if a number here looks wrong, the fix is to rerun the pipeline, not edit this file.

## Datasets and their status

| Dataset | Status | Why |
|---|---|---|
| KIMORE ex5 (squat) | **closed** | 9+ independent strategies (regression, relative/pairwise, classification, MLP), all null. `cTS` is a per-subject general clinical score, identical across exercises, not exercise-specific. |
| UI-PRMD (deep squat) | **indicative** | Strong raw signal, but the released `Data_Correct/Incorrect.csv` has no subject id (can't rule out leakage), and its `quality_score` label is a GMM log-likelihood of similarly-reduced data (partly circular). |
| REHAB24-6 | **active** | 6 exercises, 2D pose keypoints, subject id + correct/incorrect per rep in the filename - the only dataset here where a proper subject-grouped nested CV (no subject in both train and test) is actually possible. |

## Model
- Task: correct/incorrect classification (REHAB24, UI-PRMD) and quality-score regression (KIMORE, UI-PRMD)
- Type: trajectory-PCA or hand-picked kinematic features -> nested-CV model (dummy / ridge or
  logreg / random forest / MLP), see `src/quality_model.py`
- Anti-leakage: `--groups <col>` makes both the outer and inner CV loop group-aware
  (Group/StratifiedGroupKFold) - required whenever a subject contributes more than one row

## Data
- KIMORE: Kinect v2 skeleton (25 joints, position+orientation/frame), `src/build_features_ex5.py`
- UI-PRMD: Vicon, already resampled to 117 frames x 240 dims (units/joint mapping not published
  for this specific file), `src/build_features_ui_prmd.py`
- REHAB24-6: 2D pose keypoints (26 points), `src/build_features_rehab24.py`
  (whole-rep trajectory PCA) and `src/build_features_rehab24_phases.py` (early/mid/late
  thirds PCA'd separately, testing whether phase-locating the signal helps - see notebook 05)

## Performance
See `notebooks/05_experiments_analysis.ipynb` for the full table (every dataset x model, mean,
std, and a 95% bootstrap CI over outer folds) and plots, including:
- the direct comparison of `rehab24_ex1` (correct, subject-grouped split) against
  `rehab24_ex1_ungrouped` (same data, split per row) - the concrete illustration of why the
  anti-leakage rule matters, not just an assertion
- AUC per REHAB24 exercise, to see how broadly the signal holds
- whether phase-segmented features (`rehab24_ex1_phases`) change anything

## Interpretability
- SHAP: not run yet - waiting until REHAB24's signal is confirmed stable across exercises
  (Fase 4 in the playbook is downstream of a working model, which KIMORE never produced)
- Per-phase deviation from reference: `build_features_rehab24_phases.py` is a first, rule-based
  step in this direction (see "Le fasi del movimento aiutano?" in notebook 05)

## Intended use & limitations
- Research / portfolio only. Not medical advice, not a substitute for a physiotherapist or a
  qualified Pilates instructor.
- REHAB24's 26 keypoints have no published anatomical mapping alongside this specific file, so
  current REHAB24 features are generic trajectory shape (PCA), not named joint angles - a real
  limitation for the project's own interpretability thesis, to close before Fase 4.
- Self-recorded Pilates data (Layer 2) not started.

## Ethics
- Public datasets used under their data-use terms. Self-recorded clips only with consent.
