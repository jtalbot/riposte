N <- 100000L
D <- 50L

library(clusterGeneration)
library(MASS)
cov.matrix <- genPositiveDefMat(D, ratioLambda=100)
a <- mvrnorm(N, rep(0,D), cov.matrix$Sigma)
write.table(as.vector(a), "../pca.txt",row.names=FALSE,col.names=FALSE)
