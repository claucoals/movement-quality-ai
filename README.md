# Movement-Quality AI

Movement-quality assessment from pose keypoints: interpretable kinematic features (joint
angles, range of motion, symmetry, tempo) feeding nested-cross-validated models, with SHAP
attribution to check what the model actually relies on.

## Data

| Dataset | Content | Task |
|---|---|---|
| KIMORE | 75 subjects, ex5 (squat), Kinect skeleton, clinical score | Regression - see MODEL_CARD |
| UI-PRMD | Deep squat, correct/incorrect reps, Vicon | Classification - see MODEL_CARD |
| REHAB24-6 | 6 exercises, 2D pose keypoints, subject id + correct/incorrect per rep | Classification, subject-grouped nested CV |

Raw data is not committed (`.gitignore` excludes `data/` and `results/` entirely).

## Pipeline

```
config.yaml            -> declares every dataset to run (features file, target, task, groups)
src/run_experiments.py -> reads config.yaml, runs nested CV (src/quality_model.py) on each
                           dataset, appends every fold's result to results/experiments.csv
notebooks/07_experiments_analysis.ipynb -> reads results/experiments.csv, plots and tables
```

```
python src/run_experiments.py                 # run everything in config.yaml
python src/run_experiments.py --only ex1,ex2   # re-run just these dataset entries
```

Every number in the notebooks is read from `results/experiments.csv` or computed live -
nothing is hand-typed.

## Repository structure

```
movement-quality-ai/
├── config.yaml                        # experiment grid: one entry per dataset
├── requirements.txt
├── notebooks/
│   ├── 01_eda_kimore.ipynb
│   ├── 02_feature_check_ex5.ipynb
│   ├── 03_pairwise_relative_check.ipynb
│   ├── 04_classification_check.ipynb
│   ├── 05_eda_ui_prmd.ipynb
│   ├── 06_eda_rehab24.ipynb
│   ├── 07_experiments_analysis.ipynb
│   └── 08_shap_anatomical.ipynb
├── src/
│   ├── data_access.py                        # KIMORE / UI-PRMD access instructions
│   ├── pose_to_features.py                   # video -> joint angles -> features (MediaPipe)
│   ├── build_features_ex5.py                 # KIMORE ex5 kinematic features
│   ├── build_features_ui_prmd.py             # UI-PRMD trajectory-PCA features
│   ├── build_features_rehab24.py             # REHAB24-6 position-only trajectory PCA
│   ├── build_features_rehab24_dynamics.py    # + velocity/acceleration trajectory PCA
│   ├── build_features_rehab24_anatomical.py  # named joint-angle features
│   ├── build_features_rehab24_phases.py      # equal-thirds phase segmentation
│   ├── build_features_rehab24_biophases.py   # turnaround-based phase segmentation
│   ├── build_reference_deviation.py          # deviation from correct-rep reference trajectory
│   ├── univariate_check.py                   # per-feature correlation vs target, no CV
│   ├── quality_model.py                      # nested CV, group-aware
│   ├── run_shap.py                           # out-of-fold SHAP attribution
│   └── run_experiments.py                    # main: runs config.yaml, writes the CSV
└── MODEL_CARD.md                             # dataset status, performance, limitations
```

## References

- Capecci et al. (2019) *The KIMORE Dataset*, IEEE TNSRE
- Liao, Vakanski, Xian (2020) *A Deep Learning Framework for Assessing Physical Rehabilitation Exercises*, IEEE TNSRE
- Parmar & Morris (2019) *What and How Well You Performed: AQA*, CVPR
- Fieraru et al. (2021) *AIFit: Automatic 3D Human-Interpretable Feedback for Fitness*, CVPR
- Yan, Xiong, Lin (2018) *Spatial-Temporal GCN for skeleton action recognition*, AAAI
- Uhlrich et al. (2023) *OpenCap*, PLOS Computational Biology
