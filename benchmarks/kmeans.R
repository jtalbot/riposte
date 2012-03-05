
N <- 100000L
K <- 5L

#library(MASS)
#
#a <- rbind(
#	mvrnorm(N,c(0,0), matrix(c(1,0,0,1), 2,2)),
##	mvrnorm(N,c(4,0), matrix(c(0.5,0,0,0.5), 2,2)),
##	mvrnorm(N,c(0,4), matrix(c(1,0,0,1), 2,2)),
##	mvrnorm(N,c(2,2), matrix(c(0.25,0.2,0.2,0.25), 2,2)),
##	mvrnorm(N,c(1,0), matrix(c(0.25,0.22,0.22,0.25), 2,2))
#	)
#write.table(as.vector(a), "/Users/jtalbot/riposte/benchmarks/kmeans.txt", col.names=FALSE, row.names=FALSE)

a <- read.table("/Users/jtalbot/riposte/benchmarks/kmeans.txt")
dim(a) <- c(N*5,2)
#print(a)

means <- list(
	a[1,],
	a[2,],
	a[3,],
	a[4,],
	a[5,]
	)
print(means[[1]],"\n")
print(means[[2]],"\n")
print(means[[3]],"\n")
print(means[[4]],"\n")
print(means[[5]],"\n")


# assignment step
assignment <- function(data, means) {
	min.value <- Inf
	min.index <- 0L
	for(k in 1L:length(means)) {
		d2 <- 0
		for(j in 1L:ncol(data)) {
			d2 <- d2 + (data[,j]-means[[k]][[j]])^2
		}
		lt <- d2 < min.value
		min.index <- ifelse(lt, k, min.index)
		min.value <- ifelse(lt, d2, min.value)
	}
	min.index
}

update.means <- function(data, index) {
	d1 <- mean(split(data[,1L], index-1L, K))
	d2 <- mean(split(data[,2L], index-1L, K))
	for(i in 1L:K) {
		means[[i]] <<- c(d1[[i]], d2[[i]])
	}
}

for(i in 1:100) {
i <- assignment(a, means)
update.means(a, i)
print(means[[1]],"\n")
print(means[[2]],"\n")
print(means[[3]],"\n")
print(means[[4]],"\n")
print(means[[5]],"\n")
print("\n")
}


#colSum <- function(data) {
#	result <- 0
#	if(nrow(data) >= ncol(data)) {
#		for(d in 1L:ncol(data)) {
#			result[d] <- sum(data[,d])
#		}
#		# or
#		#as.vector(lapply(1L:ncol(data), function(d) sum(data[,d])))
#	} else {
#		for(d in 1L:nrow(data)) {
#			result <- result + data[d,]
#		}
#	}
#	result
#}

#update.means <- function(data, index) {
#	for(k in 1L:K) {
#		means[[k]] <- colSum(data[k==index,])
#	}
	# or
	#lapply((1L:K), function(k) colSum(data[k==index,]))
#}

## handle nested lapplys
##	lapply handles reductions where computation is perpindicular to result
##	=> result is short compared to computation
##	=> could unroll
##	=> how to distinguish from dynamic lapply?
##	=> for loop implies loop carried dependencies
##		lapply doesn't

#kmeans <- function(data, means) {
#	for(i in 1:100) {
#		means <- update.means(data, assignment(data, means))
#	}
#	means
#}

#kmeans(a, means)

