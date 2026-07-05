# Model Card

## How to read this card
Every number here is reproducible: `python src/run_experiments.py` regenerates
`results/experiments.csv` from `config.yaml`, and `notebooks/07_experiments_analysis.ipynb`
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
- REHAB24-6: 2D pose keypoints (26 points, mocap-derived, mapping confirmed via the dataset's
  Zenodo record, `data/ui_prmd/joints_names.txt`), three feature families all built per exercise
  and compared head-to-head: `src/build_features_rehab24.py` (whole-rep position-only trajectory
  PCA, "base"), `src/build_features_rehab24_dynamics.py` (position+velocity+acceleration
  trajectory PCA, "dynamics"), `src/build_features_rehab24_anatomical.py` (named joint angles -
  knee, hip, elbow, shoulder, ankle, trunk flexion, knee-valgus proxy, rep tempo, "anatomical").
  `src/build_features_rehab24_phases.py` (early/mid/late thirds PCA'd separately) tests phase
  segmentation on top of the base family.

## Performance
See `notebooks/07_experiments_analysis.ipynb` for the full table (every dataset x model, mean,
std, and a 95% bootstrap CI over outer folds) and plots, including:
- the direct comparison of `rehab24_ex1` (correct, subject-grouped split) against
  `rehab24_ex1_ungrouped` (same data, split per row) - the concrete illustration of why the
  anti-leakage rule matters, not just an assertion (ungrouped AUC 0.988 vs grouped 0.902, i.e.
  the leakage inflates the estimate, it doesn't just add noise)
- a base vs dynamics vs anatomical comparison across all 6 exercises: `dynamics` gives the best
  discriminative AUC on 4/6 exercises, `anatomical` wins the remaining 2 (ex2, ex6) - despite
  anatomical showing the broadest univariate signal of any family on every exercise in an earlier
  check, that didn't fully translate into the best multivariate model at this sample size
- `rehab24_pooled_leave_one_exercise_out`: anatomical features, trained on 5 exercises and
  tested on a 6th never seen, as a small proxy for cross-exercise transfer. Signal is present
  but modest (rf's 95% CI clears the dummy's, logreg/mlp's don't) - an early hint that "correct
  execution" isn't fully exercise-specific, not proof of robust transfer to a new domain
- whether phase-segmented features (`rehab24_ex1_phases`) change anything (no: 0.772 vs 0.902
  for the base family on ex1, so phase segmentation costs signal here, at least in this simple
  rule-based form)

## Interpretability
- SHAP: `src/run_shap.py` computes out-of-fold permutation SHAP (same grouped-CV anti-leakage
  split as the main sweep - every sample explained only by a model that never trained on it) for
  each exercise's winning anatomical model, read from `results/experiments.csv` rather than
  hardcoded. See `notebooks/08_shap_anatomical.ipynb` for the full per-exercise ranking and
  beeswarm plots. Headline finding: on Ex6 (squat), `knee_valgus_min` - a feature built
  specifically to capture knee cave-in, a known clinical squat-quality marker - is the single
  most important feature by SHAP, without ever being told that's what to look for. Across
  exercises, though, the top features are mostly exercise-specific (no feature appears in more
  than 2/6 exercises' top-5) - honestly, not yet a shared movement-quality vocabulary.
- Known limitation, stated plainly rather than glossed over: `rep_dur_mean`/`rep_dur_std` in
  `build_features_rehab24_anatomical.py` were meant as a movement-tempo proxy, but since each
  file is already a single repetition, `find_peaks` frequently can't find the 2+ troughs it
  needs and the feature comes back empty for many reps (visible as the imputer's "skipping
  features without any observed values" warning during `run_shap.py`). This is an extraction
  problem, not a weak signal - the feature should be treated as currently unreliable, not
  low-importance, until it's recomputed from multi-rep sequences instead of per-rep files.
- Per-phase deviation from reference: `build_features_rehab24_phases.py` (early/mid/late thirds)
  was the first attempt at this and didn't improve accuracy (see notebook 07) - the phase
  boundaries themselves were the likely problem (naive time-thirds, not biomechanical events),
  which is what phase segmentation from movement kinematics (e.g. pelvis vertical velocity for
  squat-like exercises) is meant to fix next.

## Intended use & limitations
- Research use only. Not medical advice, not a substitute for a physiotherapist or a
  qualified Pilates instructor.
- The anatomical family's strong univariate signal doesn't consistently win the multivariate
  comparison - a real, honestly-reported gap between "this feature correlates with the label" and
  "this feature set is the best classifier input" at REHAB24's small per-exercise sample sizes.
- Self-recorded Pilates data collection not started.

## Ethics
- Public datasets used under their data-use terms. Self-recorded clips only with consent.
