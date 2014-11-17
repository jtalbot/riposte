
pmin <- pmin.int <- function(na.rm, ...) {
    args <- list(...)
    
    if(length(args) == 0L)
        .stop("no arguments")

    r <- NULL
    for(i in seq_along(args)) {
        if(is.null(r))
            r <- strip(args[[i]])
        else
            r <- pmin(r, strip(args[[i]]))
    }

    r
}

pmax <- pmax.int <- function(na.rm, ...) {
    args <- list(...)
    
    if(length(args) == 0L)
        .stop("no arguments")

    r <- NULL
    for(i in seq_along(args)) {
        if(is.null(r))
            r <- strip(args[[i]])
        else
            r <- pmax(r, strip(args[[i]]))
    }

    r
}

