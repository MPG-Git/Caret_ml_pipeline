# 00_functions.R
# Reusable helpers for the pipeline

suppressPackageStartupMessages({
  library(caret)
  library(dplyr)
  library(readr)
  library(purrr)
  library(pROC)
})

#-------------------------------
# I/O & utility
#-------------------------------
ensure_dir <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE)
  invisible(path)
}

timestamp <- function() format(Sys.time(), "%Y%m%d_%H%M%S")

safe_write_csv <- function(x, path) {
  readr::write_csv(x, path)
  message(sprintf("Wrote: %s", path))
}

#-------------------------------
# Data helpers
#-------------------------------
map_class_labels <- function(y, labels = NULL) {
  # Converts numeric or character class column to factor with generic labels
  y <- as.vector(y)
  u <- sort(unique(y))
  k <- length(u)
  if (is.null(labels)) {
    labels <- paste0("Class", LETTERS[seq_len(k)])
  } else if (length(labels) < k) {
    stop("Provided labels fewer than unique classes.")
  }
  # Map existing levels to labels in order of appearance/sort
  fac <- factor(y, levels = u, labels = labels[seq_len(k)])
  fac
}

auto_predictors <- function(df, class_col, id_col = NULL) {
  # Choose numeric predictors by default (excluding id, class)
  drops <- c(class_col, id_col)
  cn <- setdiff(colnames(df), drops)
  keep <- cn[sapply(df[, cn, drop = FALSE], is.numeric)]
  keep
}

build_formula <- function(class_col, predictors) {
  as.formula(paste(class_col, "~", paste(predictors, collapse = " + ")))
}

#-------------------------------
# Splits
#-------------------------------
make_split <- function(df, class_col, p_train = 0.7, seed = 1L) {
  set.seed(seed)
  idx <- caret::createDataPartition(df[[class_col]], p = p_train, list = FALSE)
  list(
    train = df[idx, , drop = FALSE],
    test  = df[-idx, , drop = FALSE]
  )
}

#-------------------------------
# Train control
#-------------------------------
make_train_control <- function(cv_folds = 10, cv_repeats = 5, sampling = c("none","up","down","smote"),
                               allow_parallel = TRUE) {
  sampling <- match.arg(sampling)
  summary_fun <- if (exists("multiClassSummary", where = asNamespace("caret"), inherits = FALSE)) {
    caret::multiClassSummary
  } else {
    caret::defaultSummary
  }
  ctrl <- caret::trainControl(
    method = "repeatedcv",
    number = cv_folds,
    repeats = cv_repeats,
    classProbs = TRUE,
    savePredictions = "final",
    summaryFunction = summary_fun,
    allowParallel = allow_parallel
  )
  if (sampling != "none") ctrl$sampling <- sampling
  ctrl
}

#-------------------------------
# Model training
#-------------------------------
safe_train <- function(...) caret::train(...)

train_models <- function(formula, data, ctrl, tune_length = 3, methods = c("RF","SVM","XGB","NNET","C50","LOG")) {
  out <- list()
  # Map friendly names to caret method names + typical preprocessing
  method_map <- list(
    RF   = list(method = "rf",       args = list(importance = TRUE)),
    SVM  = list(method = "svmRadial",args = list(preProcess = c("center","scale"))),
    XGB  = list(method = "xgbTree",  args = list()), # keep generic; tuneLength controls grid breadth
    NNET = list(method = "nnet",     args = list(trace = FALSE)),
    C50  = list(method = "C5.0",     args = list()),
    LOG  = list(method = "multinom", args = list(trace = FALSE))
  )
  for (m in methods) {
    if (!m %in% names(method_map)) stop("Unknown model: ", m)
    spec <- method_map[[m]]
    if (m == "XGB" && utils::packageVersion("xgboost") >= "3.0.0") {
      spec$method <- xgb_caret_model()
    }
    args <- c(list(form = formula, data = data, method = spec$method,
                   trControl = ctrl, tuneLength = tune_length), spec$args)
    fit <- do.call(safe_train, args)
    out[[m]] <- fit
  }
  out[!vapply(out, is.null, logical(1))]
}

#-------------------------------
# Overfit/underfit detector
#-------------------------------
detect_overfit_underfit <- function(model, model_name,
                                    overfit_diff = 0.10,
                                    underfit_acc = 0.45,
                                    min_train_acc = 0.70) {
  if (is.null(model) || is.null(model$results) || !"Accuracy" %in% names(model$results)) return(NULL)
  train_acc <- mean(predict(model, newdata = model$trainingData) == model$trainingData$.outcome)
  chosen <- merge(model$results, model$bestTune, by = names(model$bestTune))
  cv_acc <- chosen$Accuracy[[1]]
  diff <- train_acc - cv_acc
  status <- if (diff > overfit_diff && train_acc > min_train_acc) {
    "Overfitting"
  } else if (cv_acc < underfit_acc) {
    "Underfitting"
  } else {
    "OK"
  }
  tibble::tibble(Model = model_name,
                 Train_Accuracy = round(train_acc, 4),
                 CV_Accuracy    = round(cv_acc, 4),
                 Difference     = round(diff, 4),
                 Status         = status)
}

#-------------------------------
# External evaluation
#-------------------------------
safe_auc <- function(truth, probs) {
  # Multi-class AUC via pROC::multiclass.roc; binary uses roc
  if (is.null(probs)) return(NA_real_)
  truth <- as.factor(truth)
  lv <- levels(truth)
  if (length(lv) == 2) {
    # Expect a single column prob for positive class
    pos <- lv[2]
    if (is.data.frame(probs) || is.matrix(probs)) {
      if (pos %in% colnames(probs)) {
        pr <- probs[, pos]
      } else {
        pr <- probs[, 1]
      }
    } else {
      pr <- probs
    }
    auc <- tryCatch(pROC::roc(truth, pr, levels = lv, direction = "<")$auc, error = function(e) NA_real_)
    as.numeric(auc)
  } else {
    auc <- tryCatch(as.numeric(pROC::multiclass.roc(truth, as.matrix(probs))$auc),
                    error = function(e) NA_real_)
    auc
  }
}

evaluate_external <- function(models, test_df, class_col) {
  res <- purrr::imap(models, function(mod, name) {
    preds <- tryCatch(predict(mod, newdata = test_df), error = function(e) NA)
    probs <- tryCatch(predict(mod, newdata = test_df, type = "prob"), error = function(e) NULL)
    if (all(is.na(preds))) return(NULL)
    cm <- caret::confusionMatrix(preds, test_df[[class_col]])
    tibble::tibble(
      Model = name,
      External_Accuracy = as.numeric(cm$overall["Accuracy"]),
      External_Kappa    = as.numeric(cm$overall["Kappa"]),
      External_AUC      = safe_auc(test_df[[class_col]], probs)
    )
  })
  dplyr::bind_rows(res)
}

summarize_external <- function(tbl) {
  tbl %>%
    dplyr::group_by(Model) %>%
    dplyr::summarise(
      Mean_External_Accuracy = mean(External_Accuracy, na.rm = TRUE),
      SD_External_Accuracy   = sd(External_Accuracy, na.rm = TRUE),
      Accuracy_CI_Lower = Mean_External_Accuracy - 1.96 * SD_External_Accuracy / sqrt(dplyr::n()),
      Accuracy_CI_Upper = Mean_External_Accuracy + 1.96 * SD_External_Accuracy / sqrt(dplyr::n()),
      Mean_External_Kappa    = mean(External_Kappa, na.rm = TRUE),
      SD_External_Kappa      = sd(External_Kappa, na.rm = TRUE),
      Kappa_CI_Lower = Mean_External_Kappa - 1.96 * SD_External_Kappa / sqrt(dplyr::n()),
      Kappa_CI_Upper = Mean_External_Kappa + 1.96 * SD_External_Kappa / sqrt(dplyr::n()),
      Mean_External_AUC      = mean(External_AUC, na.rm = TRUE),
      SD_External_AUC        = sd(External_AUC, na.rm = TRUE),
      AUC_CI_Lower = Mean_External_AUC - 1.96 * SD_External_AUC / sqrt(dplyr::n()),
      AUC_CI_Upper = Mean_External_AUC + 1.96 * SD_External_AUC / sqrt(dplyr::n())
    ) %>%
    dplyr::arrange(dplyr::desc(Mean_External_AUC)) %>%
    dplyr::ungroup()
}
