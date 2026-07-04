# Movement-Quality AI — from clinical benchmarks to Pilates form feedback

**Automatic assessment of movement quality from video, using pose estimation and machine
learning — bridging clinical rehabilitation benchmarks and real Pilates practice.**

A physiotherapist or a Pilates instructor judges a movement in seconds: is the pelvis level,
is the spine over-extending, are the two sides symmetric? This project asks whether a model
can learn that judgement from body kinematics — and, crucially, explain *why* it scored a
movement the way it did, so the feedback is actionable.

The work follows the **Action Quality Assessment (AQA)** literature and is built in two layers.

## Two-layer design

**Layer 1 — Rigorous core on public data.**
Public rehab datasets (KIMORE, UI-PRMD) provide skeletons + expert quality scores /
correct-vs-incorrect labels. From the joint trajectories we engineer interpretable
kinematic features (joint angles over time, range of motion, symmetry, tempo) and train
models in **nested cross-validation** to predict the quality score (regression) or
classify execution (good/bad). Benchmarkable, reproducible, portfolio-solid.

**Layer 2 — The signature: Pilates form feedback.**
Self-recorded Pilates exercises → pose estimation (MediaPipe / MMPose) → the same feature
pipeline → automatic, interpretable feedback on execution. This is where the project stops
being generic and becomes personal.

## Research questions

1. Can interpretable kinematic features predict expert movement-quality scores on KIMORE?
2. Does a transparent feature-based model match heavier skeleton deep-learning baselines?
3. Does the approach transfer from clinical rehab movements to Pilates exercises?

## Data (public, free)

| Dataset | Content | Use |
|---|---|---|
| **KIMORE** | 78 subjects, 5 low-back exercises, RGB+depth+skeleton, expert quality scores | Quality-score regression |
| **UI-PRMD** | 10 exercises, correct & incorrect repetitions, Kinect+Vicon | Correct-vs-incorrect classification |
| **IntelliRehabDS** | Rehab movements with correctness labels | Extra validation |
| *Self-recorded Pilates* | Your own videos, a few exercises, correct + common faults | Layer 2 demo |

Raw data / videos are **not** committed. See `src/data_access.py`.

## Pipeline

```
video  ->  pose estimation (MediaPipe / MMPose)  ->  joint-angle time series
       ->  kinematic features (ROM, symmetry, tempo)  ->  nested CV
       ->  model (quality regression / good-bad classification)
       ->  interpretability (SHAP + per-phase deviation)  ->  feedback + figures
```

## Repository structure

```
movement-quality-ai/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/ROADMAP.md
├── data/                     # access scripts only
├── notebooks/                # EDA, results, Pilates demo
├── src/
│   ├── data_access.py        # KIMORE / UI-PRMD access instructions
│   ├── pose_to_features.py   # video -> joint angles -> features (MediaPipe)
│   └── quality_model.py      # nested CV: regression & classification
├── results/
└── MODEL_CARD.md
```

## Key references

- Capecci et al. (2019) *The KIMORE Dataset*, IEEE TNSRE
- Liao, Vakanski, Xian (2020) *A Deep Learning Framework for Assessing Physical Rehabilitation Exercises*, IEEE TNSRE  (UI-PRMD)
- Parmar & Morris (2019) *What and How Well You Performed: AQA*, CVPR
- Fieraru et al. (2021) *AIFit: Automatic 3D Human-Interpretable Feedback for Fitness*, CVPR
- Yan, Xiong, Lin (2018) *Spatial-Temporal GCN for skeleton action recognition*, AAAI
- Uhlrich et al. (2023) *OpenCap*, PLOS Computational Biology

## Author

Claudia — BSc Artificial Intelligence. Portfolio project toward the M.Sc. Human Technology
in Sports and Medicine (German Sport University Cologne).
