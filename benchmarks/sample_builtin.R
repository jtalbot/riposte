
benchmark <- function(n) {
	means <- c(0,2,10)
	sd <- c(1,0.1,3)

	a <- runif(n)
	i <- floor(runif(n)*3)+1L
	rnorm(n, means[i], sd[i])
}

system.time(benchmark(10000000))
