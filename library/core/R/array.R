
array <- function(data, dim, dimnames) {
	r <- strip(data)
    if(length(r) > prod(dim))
        r <- r[seq_len(prod(dim))]
	dim(r) <- strip(dim)
    dimnames(r) <- strip(dimnames)
	r
}

is.array <- function(x) UseMethod('is.array')

is.array.default <- function(x) length(attr(x,'dim')>0L)

