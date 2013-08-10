
lapply <- function(X, FUN) {
	.External('mapply', list(X), FUN)
}

vapply <- function(X, FUN, FUN.VALUE, USE.NAMES) {
    .stop("vapply (NYI)")
}

