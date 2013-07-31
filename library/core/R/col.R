
col <- function(d) {
    if(length(d) != 2L)
        .stop("a matrix-like object is required as an argument to 'col'")

    m <- index(d[[2L]], d[[1L]], d[[1]]*d[[2]])
    dim(m) <- d
    m
}

