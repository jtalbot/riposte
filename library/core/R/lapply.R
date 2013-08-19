
lapply <- function(X, FUN) {
	.External('mapply', list(X), FUN)
}

vapply <- function(X, FUN, FUN.VALUE, USE.NAMES) {
    types <- .Map(function(x) .type(x), list(FUN.VALUE), 'character')[[1L]]
    r <- .Map(FUN, list(X), types)[[1L]]
    n <- names(X)
    if(!is.null(X))
        names(r) <- n
    else if(USE.NAMES && is.character(X))
        names(r) <- X
    r
}

