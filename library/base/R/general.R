
force <- function(x) x

mapply <- function(FUN, ...) {
	lapply(t.list(...), FUN)
}

paste <- function(..., sep = " ", collapse = NULL) {
	r <- mapply(function(x) .Internal(paste)(x, sep), ...)
	if(!is.null(collapse)) .Internal(paste)(r, collapse)
	else r
}
