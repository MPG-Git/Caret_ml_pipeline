# 01_install_libs.R
# Lightweight installer/loader for required packages (caret + common learners).
# Run once per machine (or add to CI).

pkgs <- c(
  "caret", "e1071", "nnet", "randomForest", "xgboost", "C50",
  "pROC", "PRROC",
  "smotefamily", "ROSE",
  "dplyr", "readr", "tidyr", "purrr", "ggplot2",
  "doParallel", "foreach", "parallel",
  "openxlsx", "writexl",
  "optparse"
)

install_if_missing <- function(pkgs) {
  ip <- rownames(installed.packages())
  for (p in pkgs) {
    if (!p %in% ip) {
      message(sprintf("Installing: %s", p))
      install.packages(p, dependencies = TRUE)
    } else {
      message(sprintf("Already installed: %s", p))
    }
  }
}

install_if_missing(pkgs)

invisible(lapply(pkgs, function(p) library(p, character.only = TRUE)))
message("✅ All libraries installed and loaded.")
