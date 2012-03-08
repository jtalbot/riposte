
#data <- sample(1:100, 100000, replace=TRUE)

data <- as.integer(runif(10000000,0,100))
force(data)

benchmark <- function() {
	#length(split(data,data,100L))
	tabulate(data,100L)
}

system.time(benchmark())
