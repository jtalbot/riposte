
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

typeof <- function(x) {
    switch(.type(x),
        'character' = 
            ifelse(any(attr(x,'class')=='name'), 'symbol', 'character'),
        'list' =
            ifelse(any(attr(x,'class')=='call'), 'language',
            ifelse(any(attr(x,'class')=='expression'), 'expression', 'list')),
        .type(x)
    )
}
