# Run from any working directory: Rscript /path/to/tests/smoke.R
args <- commandArgs(FALSE)
root <- dirname(dirname(normalizePath(sub("^--file=", "", grep("^--file=", args, value=TRUE)[[1]]))))
out <- tempfile("caret-smoke-")
dir.create(out)
script <- file.path(root,"R","02_train_models.R")
command <- c(shQuote(script), "--data", shQuote(file.path(root,"data","dummy_data.csv")),
             "--outdir", shQuote(out), "--models", "LOG,RF,SVM,NNET,C50,XGB",
             "--resampling", "up,down,smote", "--cv-folds", "3", "--cv-repeats", "1",
             "--tune-length", "1", "--repeats", "1", "--cores", "1")
status <- system2(file.path(R.home("bin"),"Rscript"), command, stdout=file.path(out,"stdout.log"), stderr=file.path(out,"stderr.log"))
stopifnot(status==0)
result <- read.csv(file.path(out,"External_Validation_All_Runs.csv"))
stopifnot(nrow(result)==18, all(is.finite(result$External_Accuracy)),
          all(result$External_Accuracy>=0 & result$External_Accuracy<=1),
          all(is.finite(result$External_AUC)))
fit <- read.csv(file.path(out,"Overfitting_Detection_Report.csv"))
stopifnot(nrow(fit)==18, all(is.finite(fit$Train_Accuracy)), all(is.finite(fit$CV_Accuracy)))
bad <- command; bad[which(bad=="LOG,RF,SVM,NNET,C50,XGB")] <- "UNKNOWN"
status <- suppressWarnings(system2(file.path(R.home("bin"),"Rscript"), bad, stdout=FALSE, stderr=FALSE))
stopifnot(status!=0)
source(file.path(root,"R","00_functions.R"))
source(file.path(root,"R","xgb_compat.R"))
data <- read.csv(file.path(root,"data","dummy_data.csv"))
data$class <- factor(ifelse(data$class==data$class[1],"A","B"))
binary <- train_models(build_formula("class",auto_predictors(data,"class","id")),
                       data,make_train_control(3,1,"up",FALSE),1,"XGB")[[1]]
probs <- predict(binary,newdata=data,type="prob")
stopifnot(ncol(probs)==2, all(abs(rowSums(probs)-1)<1e-6), all(is.finite(as.matrix(probs))))
cat("PASS: 18 external-validation results; unknown models fail. Artifacts:",out,"\n")
