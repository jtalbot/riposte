library(MASS)
a <- rbind(
	mvrnorm(100,c(0,0), matrix(c(1,0,0,1), 2,2)),
	mvrnorm(100,c(4,0), matrix(c(0.5,0,0,0.5), 2,2)),
	mvrnorm(100,c(0,4), matrix(c(1,0,0,1), 2,2)),
	mvrnorm(100,c(2,2), matrix(c(0.25,0.2,0.2,0.25), 2,2)),
	mvrnorm(100,c(1,0), matrix(c(0.25,0.22,0.22,0.25), 2,2))
	)
write.table(as.vector(a), "../kmeans.txt",row.names=FALSE,col.names=FALSE)

