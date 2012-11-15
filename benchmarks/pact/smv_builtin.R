
# sparse matrix-vector multiplication
# TODO: sort input by row for perf?

# random 1M x 1M matrix with 10M entries
library(Matrix)

m <- sparseMatrix(
	as.integer(runif(10000000, 1, 1000000)),
	sort(as.integer(runif(10000000, 1, 1000000))),
	x=runif(10000000),
	dims=c(1000000,1000000))

v <- runif(1000000)

smv <- function(m, v) {
	m %*% v
}

system.time(smv(m,v))
