
cbind <- function(deparse.level, ...) {
	l <- list(...)
	rows <- max(unlist(lapply(l, length),FALSE,FALSE))
	matrix <- unlist(l,FALSE,FALSE)
	dim(matrix) <- c(rows, length(l))
	matrix
}

