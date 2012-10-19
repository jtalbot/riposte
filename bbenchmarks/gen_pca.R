
# pca (via eigen values of covariance matrix) + reprojection onto new basis

{

N <- as.integer(commandArgs(TRUE)[[1]])
D <- 50L

library(MASS)
library(clusterGeneration)

cov.matrix <- genPositiveDefMat(D, ratioLambda=100)
a <- mvrnorm(N, rep(0,D), cov.matrix$Sigma)
write.table(as.vector(a), "data/pca.txt",row.names=FALSE,col.names=FALSE)

}
