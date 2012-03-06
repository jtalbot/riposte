
N <- 50000
D <- 30

library(clusterGeneration)
library(MASS)
cov.matrix <- genPositiveDefMat(D-1, ratioLambda=100)

m <- scale(mvrnorm(N, rep(0,D-1), cov.matrix$Sigma))
m <- cbind(rep(1,N),m)
w <- rnorm(D)
r <- as.double(((m %*% w) + rnorm(N,0,10)) > 0)

write.table(as.vector(m), "data/lr_p.txt", row.names=FALSE, col.names=FALSE)
write.table(as.vector(r), "data/lr_r.txt", row.names=FALSE, col.names=FALSE)
write.table(as.vector(w), "data/lr_w.txt", row.names=FALSE, col.names=FALSE)
write.table(as.vector(rnorm(D)), "data/lr_wi.txt", row.names=FALSE, col.names=FALSE)
