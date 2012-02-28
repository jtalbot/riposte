
# sparse matrix-vector multiplication

smv <- function(m, v) {
	sum(split(m$val*v[m$col], m$row-1, length(v)))
}

m <- list(
	row=c(1,2,2,3,4),
	col=c(1,1,2,3,4),
	val=c(2.5,1.5,1,9.5,1)
	)

v <- c(1,2,3,4)

