# 02_train_models.R
# Batchable training script with external validation & reporting.
# Usage:
# Rscript R/02_train_models.R --data data/input.csv --class-col class --id-col id --outdir results --predictors auto

suppressPackageStartupMessages({
  library(optparse)
  library(readr)
  library(dplyr)
  library(caret)
  library(doParallel)
})

script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_dir <- dirname(normalizePath(sub("^--file=", "", script_arg[[1]])))
source(file.path(script_dir, "00_functions.R"))
source(file.path(script_dir, "xgb_compat.R"))

# ------------------------
# CLI options
# ------------------------
option_list <- list(
  make_option(c("--data"), type="character", default="data/input.csv",
              help="Path to input CSV [default %default]"),
  make_option(c("--outdir"), type="character", default="results",
              help="Output directory [default %default]"),
  make_option(c("--id-col"), type="character", default="id",
              help="ID column name (excluded from predictors) [default %default]"),
  make_option(c("--class-col"), type="character", default="class",
              help="Outcome column (classification) [default %default]"),
  make_option(c("--predictors"), type="character", default="auto",
              help="Comma-separated predictor names, or 'auto' to select all numeric columns [default %default]"),
  make_option(c("--covars"), type="character", default="",
              help="Comma-separated covariate names to include as predictors [default none]"),
  make_option(c("--models"), type="character", default="RF,SVM,XGB,NNET,C50,LOG",
              help="Comma-separated model set (choices: RF,SVM,XGB,NNET,C50,LOG) [default %default]"),
  make_option(c("--resampling"), type="character", default="up,down,smote",
              help="Comma-separated resampling strategies to run (choices: up,down,smote) [default %default]"),
  make_option(c("--cv-folds"), type="integer", default=10, help="CV folds [default %default]"),
  make_option(c("--cv-repeats"), type="integer", default=5, help="CV repeats [default %default]"),
  make_option(c("--tune-length"), type="integer", default=3, help="caret tuneLength per model [default %default]"),
  make_option(c("--p-train"), type="double", default=0.7, help="Train proportion for external split [default %default]"),
  make_option(c("--repeats"), type="integer", default=5, help="Number of outer repeats with new splits [default %default]"),
  make_option(c("--cores"), type="integer", default=0, help="Cores for parallel training (0=auto) [default %default]"),
  make_option(c("--overfit-diff"), type="double", default=0.10, help="Overfit threshold: train - CV accuracy [default %default]"),
  make_option(c("--underfit-acc"), type="double", default=0.45, help="Underfit threshold: CV accuracy [default %default]"),
  make_option(c("--min-train-acc"), type="double", default=0.70, help="Min train accuracy to call overfit [default %default]"),
  make_option(c("--class-labels"), type="character", default="",
              help="Optional comma-separated class labels to use instead of defaults (ClassA,ClassB,...)")
)

opt <- parse_args(OptionParser(option_list=option_list), convert_hyphens = TRUE)
ensure_dir(opt$outdir)

# Log run parameters
run_log <- file.path(opt$outdir, paste0("run_", timestamp(), ".log"))
sink(run_log, split = TRUE)
cat("== ML Pipeline Run ==\n")
print(opt)

# ------------------------
# Data load & prep
# ------------------------
df <- readr::read_csv(opt$data, show_col_types = FALSE)

if (!opt$class_col %in% colnames(df)) stop("class_col not found in data.")
if (!is.null(opt$id_col) && !opt$id_col %in% colnames(df)) {
  message("Note: id_col not found; continuing without it."); opt$id_col <- NULL
}

# Class labels: anonymize/generic if needed
if (opt$class_labels != "") {
  lbls <- unlist(strsplit(opt$class_labels, ","))
  df[[opt$class_col]] <- map_class_labels(df[[opt$class_col]], labels = lbls)
} else {
  df[[opt$class_col]] <- map_class_labels(df[[opt$class_col]], labels = NULL)
}
cat("Class distribution:\n"); print(table(df[[opt$class_col]]))

# Predictors
covars <- if (nzchar(opt$covars)) unlist(strsplit(opt$covars, ",")) else character(0)
if (opt$predictors == "auto") {
  preds <- auto_predictors(df, class_col = opt$class_col, id_col = opt$id_col)
  preds <- union(preds, covars) # include covariates if they were non-numeric
} else {
  preds <- unique(c(unlist(strsplit(opt$predictors, ",")), covars))
}
if (length(preds) < 1) stop("No predictors selected.")
formula <- build_formula(opt$class_col, preds)
cat("Formula:\n"); print(formula)

# ------------------------
# Parallel backend
# ------------------------
cores <- opt$cores
if (cores <= 0) {
  cores <- max(1L, min(4L, parallel::detectCores() - 1L))
}
if (cores > 1) {
  cl <- parallel::makeCluster(cores)
  doParallel::registerDoParallel(cl)
  cat(sprintf("Parallel backend registered: %d cores\n", cores))
} else {
  registerDoSEQ(); cat("Running in sequential mode\n")
}

# ------------------------
# Outer repeats
# ------------------------
resampling_strategies <- unique(unlist(strsplit(opt$resampling, ",")))
if (any(!resampling_strategies %in% c("up", "down", "smote")))
  stop("Resampling must be up, down, or smote.")
models_requested <- unique(unlist(strsplit(opt$models, ",")))

all_external <- list()
all_overfit <- list()

for (i in seq_len(opt$repeats)) {
  cat(sprintf("\n==== Repeat %d of %d ====\n", i, opt$repeats))
  split <- make_split(df, class_col = opt$class_col, p_train = opt$p_train, seed = 100 + i)
  train_df <- split$train
  test_df  <- split$test

  # Controls (up/down)
  ctrl_up   <- make_train_control(opt$cv_folds, opt$cv_repeats, sampling = "up",   allow_parallel = (cores > 1))
  ctrl_down <- make_train_control(opt$cv_folds, opt$cv_repeats, sampling = "down", allow_parallel = (cores > 1))
  ctrl_base <- make_train_control(opt$cv_folds, opt$cv_repeats, sampling = "none", allow_parallel = (cores > 1))

  # Train per strategy
  if ("up" %in% resampling_strategies) {
    cat("Training with upsampling...\n")
    models_up <- train_models(formula, data = train_df, ctrl = ctrl_up,
                              tune_length = opt$tune_length, methods = models_requested)
  } else models_up <- NULL

  if ("down" %in% resampling_strategies) {
    cat("Training with downsampling...\n")
    models_down <- train_models(formula, data = train_df, ctrl = ctrl_down,
                                tune_length = opt$tune_length, methods = models_requested)
  } else models_down <- NULL

  if ("smote" %in% resampling_strategies) {
    cat("Training with SMOTE...\n")
    ctrl_smote <- make_train_control(opt$cv_folds, opt$cv_repeats, sampling = "smote", allow_parallel = (cores > 1))
    models_smote <- train_models(formula, data = train_df, ctrl = ctrl_smote,
                                 tune_length = opt$tune_length, methods = models_requested)
  } else models_smote <- NULL

  # Combine model sets
  model_sets <- list()
  if (!is.null(models_up))   for (nm in names(models_up))   model_sets[[paste0(nm, "_Up")]]   <- models_up[[nm]]
  if (!is.null(models_down)) for (nm in names(models_down)) model_sets[[paste0(nm, "_Down")]] <- models_down[[nm]]
  if (!is.null(models_smote))for (nm in names(models_smote))model_sets[[paste0(nm, "_SMOTE")]]<- models_smote[[nm]]

  model_sets <- model_sets[!vapply(model_sets, is.null, logical(1))]
  cat(sprintf("Models trained in this repeat: %d\n", length(model_sets)))
  if (length(model_sets) == 0) stop("No models trained; check model and resampling options.")

  # External evaluation
  ext_tbl <- evaluate_external(model_sets, test_df, class_col = opt$class_col)
  ext_tbl$Repeat <- i
  all_external[[i]] <- ext_tbl

  # Overfit/underfit report per model
  of_rows <- purrr::imap_dfr(model_sets, ~detect_overfit_underfit(.x, .y,
                       overfit_diff = opt$overfit_diff,
                       underfit_acc = opt$underfit_acc,
                       min_train_acc = opt$min_train_acc))
  of_rows$Repeat <- i
  all_overfit[[i]] <- of_rows
}

# Stop cluster if used
if (exists("cl")) try(parallel::stopCluster(cl), silent = TRUE)

# ------------------------
# Save reports
# ------------------------
external_combined <- dplyr::bind_rows(all_external)
overfit_combined  <- dplyr::bind_rows(all_overfit)

if (nrow(external_combined)) {
  safe_write_csv(external_combined, file.path(opt$outdir, "External_Validation_All_Runs.csv"))
  summary_tbl <- summarize_external(external_combined)
  safe_write_csv(summary_tbl, file.path(opt$outdir, "External_Validation_Summary.csv"))
} else {
  stop("External validation table is empty; no successful run to report.")
}

if (nrow(overfit_combined)) {
  safe_write_csv(overfit_combined, file.path(opt$outdir, "Overfitting_Detection_Report.csv"))
}

# Session info
capture.output(sessionInfo(), file = file.path(opt$outdir, "sessionInfo.txt"))

sink(NULL)
cat("✅ Done. See results in:", opt$outdir, "\n")
