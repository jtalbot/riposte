
# sparse matrix-vector multiplication
# TODO: sort input by row for perf?
# random 1M x 1M matrix with 100M entries
M <- 20000000L
N <- 500000L

v <- runif(N)
m <- list(
	row=force(sort(as.integer(runif(M, 1, N)))),
	col=force(as.integer(runif(M, 1, N))),
	val=force(runif(M))
	)

force(v)

f <- factor(m[[1]]-1L, (1L:N)-1L)
force(f)

smv <- function(m, v) {
	lapply(split(m[[3]]*v[m[[2]]], f), "sum")
}

system.time(smv(m,v))
