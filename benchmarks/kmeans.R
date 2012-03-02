
library(MASS)

#a <- rbind(
#	mvrnorm(100,c(0,0), matrix(c(1,0,0,1), 2,2)),
#	mvrnorm(100,c(4,0), matrix(c(0.5,0,0,0.5), 2,2)),
#	mvrnorm(100,c(0,4), matrix(c(1,0,0,1), 2,2)),
#	mvrnorm(100,c(2,2), matrix(c(0.25,0.2,0.2,0.25), 2,2)),
#	mvrnorm(100,c(1,0), matrix(c(0.25,0.22,0.22,0.25), 2,2))
#	)
a <- read.table("/Users/zdevito/riposte/benchmarks/kmeans.txt")[[1]]
dim(a) <- c(500,2)
print(a)
means <- list(
	a[1,],
	a[2,],
	a[3,],
	a[4,],
	a[5,]
	)

# assignment step
assignment <- function(data, means) {
	# explicit for loop over means avoids need to concat reductions
	# at the cost of parallelism and locality(!)
	min.value <- rep(Inf, nrow(data))
	min.index <- rep(0, nrow(data))
	for(k in 1:length(means)) {
		d2 <- rowSums((data - rep(means[[k]],each=nrow(data)))^2)
		lt <- d2 < min.value
		min.value <- ifelse(lt, d2, min.value)
		min.index <- ifelse(lt, k, min.index) 
	}
	min.index
}

split.df <- function (x, f) 
	lapply(split(x = seq_len(nrow(x)), f = f), 
    		function(ind) x[ind, , drop = FALSE])

# update mean
update.means <- function(data, index) {
	lapply(split.df(a,as.factor(index)), function(x) apply(x,2,mean))
}

kmeans <- function(data, means) {
	for(i in 1:100) {
		means <- update.means(data, assignment(data, means))
	}
	means
}

kmeans(a, means)
