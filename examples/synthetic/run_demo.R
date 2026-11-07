# Run all six classifiers with all three sampling strategies on bundled data.
fileArg <- grep("^--file=", commandArgs(FALSE), value=TRUE)[[1]]
here <- dirname(normalizePath(sub("^--file=", "", fileArg)))
repo <- normalizePath(file.path(here,"..",".."))
args <- commandArgs(trailingOnly=TRUE)
out <- if(length(args)) args[[1]] else file.path(here,"output")
if (file.exists(out) || dir.exists(out)) stop("Choose a NEW output folder")
dir.create(out,recursive=TRUE); out <- normalizePath(out)
status <- system2(file.path(R.home("bin"),"Rscript"), c(
 shQuote(file.path(repo,"R","02_train_models.R")),
 "--data", shQuote(file.path(here,"data","cohort.csv")), "--outdir",shQuote(out),
 "--models","LOG,RF,SVM,NNET,C50,XGB", "--resampling","up,down,smote",
 "--cv-folds","3", "--cv-repeats","1", "--tune-length","1", "--repeats","1", "--cores","1"))
if (status != 0) stop("Training failed; see output logs")
actual <- read.csv(file.path(out,"External_Validation_All_Runs.csv"))
stopifnot(nrow(actual)==18, length(unique(actual$Model))==18)
metrics <- c("External_Accuracy","External_AUC")
stopifnot(all(is.finite(as.matrix(actual[,metrics]))), all(as.matrix(actual[,metrics])>=0), all(as.matrix(actual[,metrics])<=1))
png(file.path(out,"accuracy.png"),width=1200,height=700)
par(mar=c(8,4,3,1)); barplot(actual$External_Accuracy,names.arg=actual$Model,las=2,ylim=c(0,1),
 col="steelblue",ylab="Held-out accuracy",main="Synthetic example: 18 model/sampling combinations")
dev.off()
reference <- read.csv(file.path(here,"expected","External_Validation_All_Runs.csv"))
stopifnot(setequal(actual$Model,reference$Model))
reference <- reference[match(actual$Model,reference$Model),]
comparison <- do.call(rbind,lapply(metrics,function(metric) {
 delta <- abs(actual[[metric]]-reference[[metric]])
 data.frame(Model=actual$Model,Metric=metric,Actual=actual[[metric]],Expected=reference[[metric]],
            AbsTolerance=.15,AbsoluteDifference=delta,Pass=delta<=.15)
}))
write.csv(comparison,file.path(out,"comparison.csv"),row.names=FALSE)
if (!all(comparison$Pass)) stop("Reference mismatch; inspect comparison.csv and the README version notes")
writeLines("PASS: 18 model/sampling combinations completed and matched reference tolerances.",file.path(out,"PASS.txt"))
cat("PASS: synthetic classification example. Results:",out,"\n")
