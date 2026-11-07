# caret's bundled xgbTree adapter writes fields into an xgb.Booster. XGBoost
# 3 uses an external pointer there; keep caret metadata in a separate list.
xgb_caret_model <- function() {
  original <- caret::getModelInfo("xgbTree", regex = FALSE)[[1]]
  probabilities <- function(modelFit, newdata, submodels = NULL) {
    raw <- predict(modelFit$booster, as.matrix(newdata))
    lev <- modelFit$obsLevels
    if (length(lev) == 2) raw <- cbind(1 - raw, raw)
    raw <- matrix(raw, ncol = length(lev))
    colnames(raw) <- lev
    as.data.frame(raw)
  }
  list(label = "XGBoost (native booster wrapper)", library = "xgboost",
       type = "Classification", parameters = original$parameters,
       grid = original$grid, sort = original$sort,
       fit = function(x, y, wts, param, lev, last, classProbs, ...) {
         matrix <- xgboost::xgb.DMatrix(as.matrix(x), label = as.integer(y) - 1L)
         if (!is.null(wts)) xgboost::setinfo(matrix, "weight", wts)
         params <- as.list(param[1, setdiff(names(param), "nrounds"), drop = FALSE])
         params$objective <- if (length(lev) == 2) "binary:logistic" else "multi:softprob"
         if (length(lev) > 2) params$num_class <- length(lev)
         params$nthread <- 1L
         booster <- xgboost::xgb.train(params = params, data = matrix,
                                      nrounds = param$nrounds, verbose = 0)
         list(booster = booster, obsLevels = lev)
       },
       prob = probabilities,
       predict = function(modelFit, newdata, submodels = NULL) {
         probs <- probabilities(modelFit, newdata)
         modelFit$obsLevels[max.col(as.matrix(probs), ties.method = "first")]
       },
       levels = function(x) x$obsLevels)
}
