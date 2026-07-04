# Model Card

## Model
- Task: movement-quality regression (KIMORE score) / correct-vs-incorrect (UI-PRMD)
- Type: kinematic features + nested-CV model
- Version / date:

## Data
- Source & split: subject-level nested CV, no subject leakage across folds
- Pose front-end: MediaPipe Pose (or provided Kinect skeletons)
- Features: joint-angle ROM / mean / std, left-right symmetry, tempo — log all settings

## Performance
- Regression: MAE, R2 (mean ± std over outer folds)
- Classification: balanced accuracy, AUC

## Interpretability
- SHAP top features:
- Per-phase deviation from reference (actionable feedback):

## Intended use & limitations
- Research / portfolio only. Not medical advice, not a substitute for a physiotherapist
  or a qualified Pilates instructor. Self-recorded data is small and not clinically validated.

## Ethics
- Public datasets used under their data-use terms. Self-recorded clips only with consent.
