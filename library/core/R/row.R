
row <- function(d) {
   if(length(d) != 2L)
        .stop("a matrix-like object is required as an argument to 'row'")

    m <- index(d[[1L]], 1L, d[[1]]*d[[2]])
    dim(m) <- d
    m
}

