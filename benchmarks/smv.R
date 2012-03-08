
# sparse matrix-vector multiplication
# TODO: sort input by row for perf?

# random 1M x 1M matrix with 10M entries
m <- list(
	row=as.integer(runif(10000000, 1, 1000000)),
	col=as.integer(runif(10000000, 1, 1000000)),
	val=runif(10000000)
	)

f <- as.factor(m[[1]]-1)

v <- runif(1000000)
force(v)

smv <- function(m, v) {
	#sum(split(m[[3]]*v[m[[2]]], m[[1]]-1, length(v)))
	#tapply(m[[3]]*v[m[[2]]], list(f), sum)
	lapply(split(m[[3]]*v[m[[2]]], f), sum)
}

system.time(smv(m,v))
