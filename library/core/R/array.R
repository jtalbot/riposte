
array <- function(data, dim, dimnames) {
	r <- strip(data)
	dim(r) <- strip(dims)
    dimnames(r) <- strip(dimnames)
	r
}

