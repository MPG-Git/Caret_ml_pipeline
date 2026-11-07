# caret-ml-pipeline (R)

A **lightweight, reproducible** machine-learning template for multi-class classification using **caret**.  
It generalizes a neuroscience-focused lab script into a **GitHub-ready** pipeline that anyone can run on their own CSV.

## Features

- Models: **RF**, **SVM (Radial)**, **XGBoost**, **Neural Net (nnet)**, **C5.0**, **Multinomial Logistic**
- Resampling: **Upsampling**, **Downsampling**, **SMOTE**
- External validation split (train/test)
- Repeated CV (configurable folds/repeats) with parallel training (optional)
- Adjustable **overfit/underfit detector** thresholds
- Clean outputs (CSV): external validation per run, summary table, overfitting report
- No private paths, no domain-specific features. Works in **relative folders** `data/` and `results/`.

## Repo layout

```
.
├─ R/
│  ├─ 01_install_libs.R   # install/load required packages
│  ├─ 00_functions.R      # reusable helpers
│  └─ 02_train_models.R   # main pipeline (training + external validation + reports)
├─ data/
│  └─ dummy_data.csv      # example dataset with predictor1..predictor10 and ClassA/B/C
├─ results/               # outputs written here
└─ README.md
```

## Input format (CSV)

- One **class/label** column (default: `class`). Can be numeric (e.g., 1/2/3) or strings—script maps to `ClassA`, `ClassB`, `ClassC` by default.
- Optional **ID** column (default: `id`).
- One or more **predictor** columns (numeric recommended). You can specify them or let the script auto-select all numeric columns (excluding id/class).

Example (dummy_data.csv):
```
id,class,predictor1,predictor2,...
1,ClassA,0.12, 1.03, ...
2,ClassB,-0.22, 0.55, ...
...
```

## Quick start

1) **Install packages (run once):**
```bash
Rscript R/01_install_libs.R
```

2) **Train & evaluate:**
```bash
# Basic: auto-predictors, defaults
Rscript R/02_train_models.R \
  --data data/dummy_data.csv \
  --outdir results

# Specify predictors and covariates
Rscript R/02_train_models.R \
  --data data/dummy_data.csv \
  --outdir results \
  --class-col class \
  --id-col id \
  --predictors predictor1,predictor2,predictor3 \
  --covars age,sex
```

3) **Tune runtime / resampling / models:**
```bash
Rscript R/02_train_models.R \
  --data data/dummy_data.csv \
  --outdir results \
  --models RF,SVM,XGB,NNET,C50,LOG \
  --resampling up,down,smote \
  --cv-folds 10 \
  --cv-repeats 5 \
  --tune-length 3 \
  --p-train 0.7 \
  --repeats 5 \
  --cores 4
```

4) **Overfitting detector thresholds (adjustable):**
```bash
Rscript R/02_train_models.R \
  --data data/dummy_data.csv \
  --outdir results \
  --overfit-diff 0.10 \
  --underfit-acc 0.45 \
  --min-train-acc 0.70
```

5) **Custom class labels (optional):**
```bash
# If your class column is numeric like 1/2/3
Rscript R/02_train_models.R \
  --data data/dummy_data.csv \
  --class-labels Foo,Bar,Baz
```

## Outputs

- `results/External_Validation_All_Runs.csv` – per-repeat, per-model metrics (Accuracy, Kappa, AUC)
- `results/External_Validation_Summary.csv` – aggregate mean/SD and Wald CIs per model
- `results/Overfitting_Detection_Report.csv` – per-model flags using your thresholds
- `results/sessionInfo.txt` – session/package versions
- `results/run_*.log` – run parameters and console log

## Notes

- This template **anonymizes** labels and uses **generic predictor names** in examples. Your real CSV can use any column names; specify them with `--predictors` or rely on `--predictors auto`.
- SMOTE, upsampling and downsampling run inside caret's training folds. SMOTE uses `themis`; external validation cases are never resampled.
- Hyperparameters: by default the script uses `tuneLength` to keep grids generic. You can swap in explicit `tuneGrid` in `train_models()` for domain-specific work.
- Excel outputs with custom formatting were removed to stay lightweight. If desired, you can write `.xlsx` via `openxlsx::write.xlsx()`.

## License

MIT
# Setup and verified workflow

This is a standalone R pipeline, not an EEGLAB plugin. Export one row per
participant to CSV, with a classification outcome and predictor columns.
Use R 4.5.1 or a compatible newer R installation. From the repository root:

```sh
Rscript R/01_install_libs.R
Rscript tests/smoke.R
Rscript R/02_train_models.R --data data/dummy_data.csv --outdir results --class-col class --id-col id --models LOG,RF,SVM,NNET,C50,XGB --resampling up,down,smote --cv-folds 3 --cv-repeats 1 --tune-length 1 --repeats 1 --cores 1
```

The bounded smoke test checks all six learners with all three sampling
strategies, validates 18 output rows, and checks a failing configuration.
The production defaults use larger repeated cross-validation grids and take
longer. `--cores 0` uses at most four workers; choose an explicit count for
larger machines. Relative data/output paths are relative to the shell's
working directory; the script's helper paths are resolved independently.

SMOTE now runs inside each training fold using caret/themis, preserving the
held-out cases and predictor units. Train accuracy is computed from training
predictions; CV accuracy comes from the chosen tuning configuration. Model
errors stop the run instead of being reported as success. XGBoost 3 is handled
through a separate booster wrapper compatible with caret's metadata.

Verified on R 4.5.1 with XGBoost 3.2.1.1. Synthetic/demo-data execution checks
software behavior, not predictive performance for a particular study. Repeated
random splits overlap, so the summary's normal-approximation interval columns
are descriptive and must not be treated as independent-sample confidence
intervals. Participant grouping, temporal splits, missing-data handling and
an independent final test cohort must be designed for the study.

---
