# Model Card

## How to read this card
Every number here is reproducible: `config.yaml` describes one experiment (one dataset from the
`datasets.yaml` catalog), `python src/run_experiments.py` runs it and appends the result to
`results/experiments/experiments.csv`, and `notebooks/07_experiments_analysis.ipynb` turns that
CSV into the tables/plots this card summarizes. Nothing below is hand-typed from a one-off run -
if a number here looks wrong, the fix is to rerun the pipeline, not edit this file. The full
36-dataset sweep and the SHAP interpretability pass both run reproducibly via GitHub Actions
(`.github/workflows/run-experiments.yml`, `run-shap.yml`) on hosted runners, not a specific
machine.

## Datasets and their status

| Dataset | Status | Why |
|---|---|---|
| KIMORE ex5 (squat) | **closed** | 9+ independent strategies (regression, relative/pairwise, classification, MLP), all null. `cTS` is a per-subject general clinical score, identical across exercises, not exercise-specific. |
| UI-PRMD (deep squat) | **indicative** | Very strong raw signal (classification AUC up to 0.99), but the released `Data_Correct/Incorrect.csv` has no subject id (can't rule out leakage), and its `quality_score` label is a GMM log-likelihood of similarly-reduced data (partly circular) - a result this strong with this little independent verification is treated as a ceiling estimate, not a claim. |
| REHAB24-6 | **active** | 6 exercises, 2D pose keypoints, subject id (from the dataset's own annotations, not the filename) + correct/incorrect per rep - the only dataset here where a proper subject-grouped nested CV (no subject in both train and test) is fully possible, and the only one with enough structure (named joints, phases, a reference-deviation signal) to support interpretability work. |

## Method
- Task: correct/incorrect classification (REHAB24, UI-PRMD) and quality-score regression
  (KIMORE, UI-PRMD).
- Models: dummy baseline, ridge/logistic regression, random forest, MLP - see
  `src/quality_model.py`. Feature selection (`SelectPercentile`) and every hyperparameter are
  searched inside the inner CV loop, never fit on data the outer fold will be scored on.
- Anti-leakage: `--groups <col>` makes both the outer and inner CV loop group-aware
  (Group/StratifiedGroupKFold) - required whenever a subject contributes more than one row.
  Concretely: `rehab24_ex1` (grouped) scores logreg at AUC 0.906; `rehab24_ex1_ungrouped` (same
  data and model, split per row instead of per subject) scores 0.981 - leakage doesn't just add
  noise, it inflates the estimate, which is why this comparison is kept in the results as a
  standing illustration rather than just an assertion in this card.
- Repeated nested CV, not single-split: with 75-116 samples a single outer split is noisy (fold
  AUCs swing by +/-0.2 between repeats on the same data), so every dataset is repeated 20 times
  with different splits and pooled before reporting mean/CI (Vabalas et al., 2019, *PLOS ONE* -
  plain k-fold stays optimistically biased at small N regardless of sample size, nested CV
  doesn't; the same small-sample literature's point about feature-selection leakage motivated
  putting `SelectPercentile` inside the pipeline rather than before it).
- Hyperparameter grids are deliberately small, not exhaustive: with ~50 samples per inner fold,
  comparing many candidates doesn't resolve a real optimum, it mostly fits inner-CV noise (Cawley
  & Talbot, 2010, *JMLR* - overfitting in model selection itself, a separate problem from the
  leakage the grouped CV already prevents). Concretely, random forest's `n_estimators` is fixed
  rather than searched (accuracy is flat above a few hundred trees, so searching it bought noise,
  not a better model, and was the single largest combinatorial driver of runtime), and the
  feature-selection percentile grid is 2 levels, not a finer sweep.
- Small-group caveat: REHAB24-6 has only 8-9 subjects per exercise. With `outer_splits=5`,
  several outer folds necessarily hold out a single subject, so any one fold's AUC is noisy by
  construction (one person's performance decides that fold) - the pooled mean/CI across 100
  outer splits (5 folds x 20 repeats) is what should be trusted, not any individual fold.
- Metrics: balanced accuracy, AUC, Brier score (calibration - are the predicted probabilities
  trustworthy, not just their ranking) and MCC (robust to the mild class imbalance some folds
  have) for classification; MAE, R^2, Spearman for regression.

## A methodological bug worth stating plainly
The dummy classifier's `random_state` was fixed across every repeat, and a fresh
`DummyClassifier(strategy="stratified", random_state=42)` was constructed on every one of the
100 outer splits. With only 8-9 subjects, the same test-fold *shape* recurs constantly across
repeats, and a freshly-seeded RNG given the same shape produces byte-identical "random"
predictions every time - so the 100 repeats were not 100 independent draws, and the dummy
baseline's reported mean drifted anywhere from 0.39 to 0.63 balanced accuracy across datasets
instead of the 0.5 a stratified-guess baseline should have. Confirmed independently in the
small-sample cross-validation literature (10-fold CV has been reported to produce AUCs as low as
0.3 on purely random data under exactly this kind of stratification bias). Fixed by varying only
the dummy's seed per repeat; every classification dataset was re-run afterward. Every dummy
baseline reported in this card now centers on 0.49-0.52, as it should.

## Data
- KIMORE: Kinect v2 skeleton (25 joints, position+orientation/frame), `src/build_features_ex5.py`.
- UI-PRMD: Vicon, already resampled to 117 frames x 240 dims (units/joint mapping not published
  for this specific file), `src/build_features_ui_prmd.py`.
- REHAB24-6: 2D pose keypoints (26 points, mocap-derived, mapping confirmed via the dataset's
  Zenodo record), five feature families built per exercise and compared head-to-head:
  - `base` - whole-rep position-only trajectory PCA
  - `dynamics` - position+velocity+acceleration trajectory PCA
  - `anatomical` - named joint angles (knee, hip, elbow, shoulder, ankle, trunk flexion,
    knee-valgus proxy) and their summary statistics, shared with the smartphone-video path
    (`src/anatomical_features.py`) so a model trained here can in principle score a phone clip
  - `biophases` - the same anatomical angles, split into descent/bottom/ascent using each rep's
    own kinematic turnaround point rather than a fixed time split
  - `deviation` - mean absolute distance from a leave-subject-out reference trajectory
    (`src/build_reference_deviation.py`), per phase and joint
  - `phases` (ex1 only) - an earlier, naive equal-thirds time split, kept as a documented
    negative comparison against `biophases` (see Performance)

## Performance
See `notebooks/07_experiments_analysis.ipynb` for the full table (every dataset x model, mean,
std, and a 95% bootstrap CI over outer folds) and plots.
- **Family comparison, REHAB24-6** (best model's AUC per exercise, all 5 families): `dynamics`
  wins 3/6 exercises (ex1 0.968, ex4 0.886, ex5 0.919), `anatomical` wins 2/6 (ex2 0.814, ex3
  0.841), `biophases` wins 1/6 (ex6 0.945). `base` and `deviation` never win outright.
  `anatomical` and `biophases` are the only families with nameable features (knee angle,
  valgus, asymmetry, by phase); `dynamics`/`base` are opaque PCA components.
- **Deviation family**: real signal on 4/6 exercises (best AUC 0.66-0.78, clearing its own
  dummy baseline) but at or below chance on the other 2 (ex2: 0.496 vs dummy 0.517; ex5: 0.515
  vs dummy 0.509), and it never wins the head-to-head against the other families on any
  exercise. 36 correlated phase x joint features against ~50-70 training samples is a plausible
  reason: SHAP-ranking stability for this family (Kendall's W across repeats, see
  Interpretability) is the lowest on average of the three interpretable families (mean W 0.58
  vs anatomical 0.68, biophases 0.73) - but not uniformly: it is the single most stable ranking
  of any family on ex1 (0.77) and ex3 (0.96), and only the outright weakest on 3 of 6 exercises.
  A tendency worth noting, not a rule; the same underlying small-N problem showing up twice is a
  plausible read, not a settled one.
- **Naive vs kinematic phase segmentation, ex1**: `base` 0.906 > `biophases` 0.850 >
  `phases` (equal-thirds) 0.833 - the naive time-split family never beats the family it was
  built on top of; turnaround-based segmentation recovers some of that gap but doesn't
  guarantee a win over not splitting into phases at all for every exercise.
- **Cross-exercise transfer** (`rehab24_pooled_leave_one_exercise_out`: train on 5 exercises,
  test on a 6th never seen, anatomical features only): random forest's mean AUC (0.626) beats
  dummy's (0.515), but with only 6 outer folds the 95% CIs are wide and do overlap (RF
  0.542-0.709 vs dummy 0.392-0.637) - a real hint that "correct execution" isn't fully
  exercise-specific, not a demonstration of robust transfer by the stricter non-overlap
  criterion used elsewhere in this card.
- **KIMORE ex5**: still null across every model (ridge R^2 -0.18, Spearman 0.09; RF and MLP
  both worse) - consistent with its `closed` status.
- **UI-PRMD**: strong (classification AUC up to 0.99, regression R^2 up to 0.65) but read
  through its `indicative` caveat above, not as a validated result.
- **MLP is consistently the weakest, least stable model** across nearly every dataset in this
  sweep, occasionally catastrophically so (UI-PRMD regression: R^2 -5.7 +/- 21.7, driven by
  specific folds landing on badly-overfit configurations). This is the expected small-N result,
  not a tuning failure - the small-sample literature above predicts exactly this, and the model
  is kept in the comparison for that reason rather than dropped.

## Interpretability
- SHAP: `src/run_shap.py` computes out-of-fold permutation SHAP (same grouped-CV anti-leakage
  split as the main sweep - every sample explained only by a model that never trained on it) for
  each exercise's winning model per family, read from `results/experiments/experiments.csv`
  rather than hardcoded. Ranking stability across the 5 repeats is reported as Kendall's W (1.0
  = every repeat agrees on the order of the top-10 features, 0 = no more agreement than chance).
- **Ex6 (squat), the closest thing to a pre-specified check**: `knee_valgus_min` was built
  specifically to capture knee cave-in, a known clinical squat-quality marker, before looking at
  SHAP results. It shows up: 5th of 83 anatomical features by SHAP, and still present (10th of
  216) once split by phase, concentrated in the descent. It is not, however, the single dominant
  feature - `l_knee_mean` (average knee flexion across the rep, a more generic depth/ROM measure)
  ranks 1st, and at the joint-aggregated level ankle and knee are close to tied (mean|SHAP| 0.152
  vs 0.150). This is reported as found: a real, present, purpose-built signal that survived three
  independent rounds of bug fixes (subject-identity leakage, mocap QC, exercise-side selection)
  and a change in feature dimensionality, but not the singular flagship result an earlier,
  less-complete pass through this data suggested.
- **Movement Attribution Map** (`src/build_attribution_map.py`, `notebooks/09_attribution_map.ipynb`):
  fuses SHAP importance and reference-deviation effect size (both as within-exercise percentile
  ranks, since raw SHAP magnitude isn't comparable across feature-family sizes) into one
  phase x joint matrix per exercise. For Ex6 specifically, fusing with deviation pulls **ankle**
  (descent and bottom) ahead of knee - ankle's deviation-from-reference effect size is very
  large (Cohen's d up to ~1.4-1.8) on top of its near-tied SHAP weight, while knee/descent
  remains clearly present (3rd overall). Two roughly-tied SHAP signals get pulled apart once
  each is checked against how far it actually deviates from correct technique.
- **Cross-exercise consistency**: mostly exercise-specific. No feature appears in more than 2 of
  6 exercises' top-5 by SHAP; the few that recur twice are ankle-velocity and trunk-flexion
  features, not a broad shared vocabulary of movement quality.
- **An observation flagged as unexplained, not glossed over**: Ex1 (arm abduction)'s dominant
  SHAP features are ankle and knee (`r_ankle_mean`, `r_ankle_max`, `r_knee_mean`), not the arm -
  no shoulder feature appears in its top-12. This doesn't match a known clinical
  trunk/shoulder-compensation pattern and isn't forced into one; it's reported as an open
  question (possibly a real postural confound, possibly a small-sample artifact) rather than
  explained away.

## Intended use & limitations
- Research use only. Not medical advice, not a substitute for a physiotherapist or a qualified
  Pilates instructor.
- Multiple comparisons: the sweep tests every feature family against every model type, per
  exercise - 6 exercises x 5 families x 4 models, each itself the product of a hyperparameter
  search. "Family X wins on N/6 exercises" summarizes the winner of that search, not a single
  pre-registered comparison, and should be read as exploratory - a place to look closer (e.g.
  with SHAP, or a held-out replication), not a confirmed result on its own. `knee_valgus_min` on
  Ex6 remains the one result closer to confirmatory, for the reasons stated above - not because
  it topped a ranking, but because it was specified before looking and has kept showing up
  since, at a modest but real rank rather than a dominant one.
- Self-recorded Pilates data collection not started; the smartphone-video path
  (`src/pose_to_features.py`) is built and schema-matched to REHAB24 but untested end-to-end
  against a real clip.
- MLP's instability (see Performance) means any single MLP result on these datasets should be
  read with more skepticism than a ridge/logreg/RF result at the same AUC.

## Ethics
- Public datasets used under their data-use terms. Self-recorded clips only with consent.
