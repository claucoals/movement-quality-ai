# Roadmap

Runs during the au-pair year as the main side-project (~mid-Oct 2026 -> summer 2027).

> Check the exact M.Sc. TSM application deadline on mySpoho (winter intake, closes in
> spring/early summer) and have the repo presentable a few weeks before.

## Phase 0 — Setup (~2 weeks)
- [ ] Push scaffold to GitHub, environment, install requirements
- [ ] Request KIMORE + UI-PRMD access
- [ ] Read core papers: Capecci (KIMORE), Liao/Vakanski (UI-PRMD), Parmar & Morris (AQA), Fieraru (AIFit)
- Deliverable: repo + one-page notes on the 4 papers

## Phase 1 — Public-data core (~6-8 weeks)
- [ ] Load KIMORE skeletons, engineer kinematic features
- [ ] Nested-CV quality-score regression (quality_model.py --task regression)
- [ ] Repeat correct-vs-incorrect on UI-PRMD (--task classification)
- Deliverable: benchmark table + ROC / error plots

## Phase 2 — Interpretability (~4-6 weeks)
- [ ] SHAP: which kinematic features drive the score
- [ ] Per-phase deviation from a reference execution (turn a number into feedback)
- Deliverable: the interpretable-feedback layer, the part that stands out

## Phase 3 — Pilates signature (~6-8 weeks)
- [ ] Record a few Pilates exercises: correct + common faults, several reps, consent
- [ ] MediaPipe -> features -> apply the Phase-1/2 pipeline
- [ ] Simple demo notebook that outputs feedback on a new clip
- Deliverable: the personal, distinctive demo for the motivation letter

## Phase 4 — Polish + outreach (~4 weeks)
- [ ] Fill MODEL_CARD, freeze README, clean notebooks
- [ ] Short technical report (4-6 pages)
- [ ] Contact the Humanitas mentor + a Cologne TSM professor with the project
- Deliverable: portfolio-ready repo; optional arXiv/medRxiv preprint
