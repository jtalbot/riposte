
cbind <- function(deparse.level, ...) {
	l <- list(...)
	rows <- max(unlist(lapply(l, length)))
	matrix <- unlist(l)
	dim(matrix) <- c(rows, length(l))
	matrix
}

