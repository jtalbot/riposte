
#data <- sample(1:100, 100000, replace=TRUE)

data <- as.integer(runif(100000000,0,100))
f <- factor(data, 0L:99L)

benchmark <- function() {
	lapply(split(data,f), "length")
	#tabulate(data,100L)
}

system.time(benchmark())
