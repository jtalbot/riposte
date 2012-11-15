
N <- 1000000L
K <- 5L

library(MASS)

a <- rbind(
	mvrnorm(N/5,c(0,0), matrix(c(1,0,0,1), 2,2)),
	mvrnorm(N/5,c(4,0), matrix(c(0.5,0,0,0.5), 2,2)),
	mvrnorm(N/5,c(0,4), matrix(c(1,0,0,1), 2,2)),
	mvrnorm(N/5,c(2,2), matrix(c(0.25,0.2,0.2,0.25), 2,2)),
	mvrnorm(N/5,c(1,0), matrix(c(0.25,0.22,0.22,0.25), 2,2))
	)
write.table(as.vector(a), "data/kmeans.txt", col.names=FALSE, row.names=FALSE)

