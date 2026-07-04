# Model Card

## Model
- Task: movement-quality regression (KIMORE cTS score) / good-bad classification (median split)
- Type: kinematic features (angles, ROM, velocity, tempo, trajectory PCA, trunk, smoothness) + nested-CV model
- Scope so far: KIMORE ex5 (squat) only, 75 subjects. UI-PRMD not yet started.
- Version / date: Fase 2 checkpoint, 2026-07-04

## Data
- Source & split: subject-level nested CV (RepeatedKFold / RepeatedStratifiedKFold, 5 outer x 10 repeats), no subject leakage across folds
- Pose front-end: KIMORE's own Kinect v2 skeleton (25 joints, position + orientation per frame)
- Features tried (all in `src/build_features_*.py`): joint-angle ROM/mean/std/velocity (knee, hip),
  left-right symmetry, squat-repetition tempo, trajectory PCA + frequency-domain descriptors,
  trunk/pelvis flexion-lean-rotation, movement smoothness (jerk/acceleration)
- Known data limitation: `cTS` is a single per-subject clinical severity score, identical across
  all 5 KIMORE exercises, not an exercise-specific execution score

## Performance (honest, Fase 2 Check-gate 2 status: RED)
Repeated nested CV (5x10=50 outer splits), DummyRegressor/DummyClassifier as baseline.

- **Regression (cTS, continuous)**: no feature family beats the dummy baseline. Best result
  (37 kinematic features): ridge MAE=0.077+/-0.014 vs dummy MAE=0.077+/-0.014, Spearman=0.120+/-0.220
  (sign unstable across folds). Adding trajectory/PCA, trunk, or smoothness features does not change this.
- **Pairwise/relative regression (CoRe-style)**: permutation-corrected family-wise test across
  57 features, p=0.49 - no signal, absolute or relative.
- **Classification (good/bad, median split of cTS)**: the one result that beats the baseline -
  rf AUC=0.622+/-0.131, logreg AUC=0.599+/-0.114 vs dummy AUC=0.512+/-0.136 (59 features:
  kinematic + trajectory/PCA). Does not survive robustness checks: adding trunk or smoothness
  features leaves AUC flat (0.56-0.63 regardless of feature set), and a sharper tertile split
  (top/bottom third, n=54) makes rf collapse to chance (AUC=0.505). Reported as a borderline,
  non-robust finding, not a working model.
- **More flexible model (MLP)**: added to rule out "wrong model class" as the explanation. In
  regression it is clearly worse than the baseline (MAE=0.113+/-0.020 vs dummy 0.077, R2 around
  -1.56: severe overfitting at n=75). In classification it sits inside the same 0.56-0.63 AUC
  plateau as logreg/rf (AUC=0.586), it does not unlock anything a simpler model was missing.
- All raw per-fold results and the univariate/permutation diagnostics behind this table are in
  `notebooks/02-05` and the background-run logs referenced in commit messages.

## Interpretability
- SHAP: not run yet (no model beats baseline convincingly enough to justify it - Fase 4 blocked
  on Fase 2/3 first).
- Per-phase deviation from reference: not started.

## Intended use & limitations
- Research / portfolio only. Not medical advice, not a substitute for a physiotherapist
  or a qualified Pilates instructor. Self-recorded data is small and not clinically validated.
- Current KIMORE model does not work well enough to use for anything beyond documenting the
  attempt and the negative result honestly.

## Ethics
- Public datasets used under their data-use terms. Self-recorded clips only with consent.
