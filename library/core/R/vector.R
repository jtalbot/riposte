
as.vector <- function(x, mode = "any") {
	switch(mode,
		logical = as(x,"logical"),
		integer = as(x,"integer"),
		double  = as(x,"double"),
		numeric = as(x,"double"),
		complex = as.complex(x),
		character = as(x,"character"),
        list = as(x,"list"),
		any = strip(x))
}

cbind <- function(...) {
	l <- list(...)
	rows <- max(unlist(lapply(l, length)))
	matrix <- unlist(l)
	dim(matrix) <- c(rows, length(l))
	matrix
}

array <- function(data, dims) {
	r <- strip(data)
	dim(r) <- strip(dims)
	r
}

seq.int <- function(from, to, by, length.out, along.with, ...) {
	seq(from, by, (to-from)/by+1)
}

seq_len <- function(length.out) seq(1, 1, length.out)
