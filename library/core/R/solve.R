
La_solve <- function(a, b, tol) {
    da <- dim(a)
    db <- dim(b)
    if(is.null(da)) {
        da <- c(length(a), 1L)
    }
    if(is.null(db)) {
        db <- c(length(b), 1L)
    }

    if(length(da) != 2)
        .stop("matrix 'A' must be 2-dimensional")
    if(da[[1L]] != da[[2L]])
        .stop("matrix 'A' must be square")

    if(da[[1L]] != db[[1L]])
        .stop(sprintf("'b' (%d x %d) must be compatible with 'a' (%d x %d)", db[[1L]], db[[2L]], da[[1L]], da[[2L]]))

    r <- .Riposte('solve', as.double(strip(a)), da[[1L]], as.double(strip(b)), db[[1L]], db[[2L]])

    if(!is.null(dim(b))) {
        dim(r) <- c(da[[1L]], db[[2L]])
    }

    r
}
