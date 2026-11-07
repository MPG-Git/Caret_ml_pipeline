# Recreate this demonstration cohort in a NEW file. No participant data.
args <- commandArgs(trailingOnly=TRUE)
fileArg <- grep("^--file=", commandArgs(FALSE), value=TRUE)[[1]]
here <- dirname(normalizePath(sub("^--file=", "", fileArg)))
target <- if (length(args)) args[[1]] else file.path(here, "data", "cohort.csv")
if (file.exists(target)) stop("Choose a new destination; bundled data will not be overwritten")
set.seed(47)
n <- 180
class <- rep(c("ClassA", "ClassB", "ClassC"), each=60)
signal <- rep(c(-1.5,0,1.5), each=60)
df <- data.frame(id=sprintf("synthetic_%03d",seq_len(n)),class=class)
for (j in 1:6) df[[paste0("feature",j)]] <- signal * (7-j)/6 + rnorm(n,sd=.8)
dir.create(dirname(target),recursive=TRUE,showWarnings=FALSE)
write.csv(df,target,row.names=FALSE)
cat(target,"\n")
