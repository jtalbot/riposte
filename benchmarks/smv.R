
# sparse matrix-vector multiplication
# TODO: sort input by row for perf?

# random 1M x 1M matrix with 10M entries
m <- list(
	row=as.integer(runif(10000000, 1, 1000000)),
	col=sort(as.integer(runif(10000000, 1, 1000000))),
	val=runif(10000000)
	)

v <- runif(1000000)
force(v)

f <- factor(m[[1]]-1L, 1L:1000000L-1L)
force(f)

smv <- function(m, v) {
	lapply(split(m[[3]]*v[m[[2]]], f), "sum")
}

system.time(smv(m,v))
