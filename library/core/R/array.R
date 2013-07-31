
array <- function(data, dim, dimnames) {
	r <- strip(data)
	dim(r) <- strip(dims)
    dimnames(r) <- strip(dimnames)
	r
}

is.array <- function(x) UseMethod('is.array')

is.array.default <- function(x) length(attr(x,'dim')>0L)

