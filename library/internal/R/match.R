
.match <- function(x, table) {
    if(is.list(x) || is.list(table))
        index <- .semijoin(as.character(x), as.character(table))
    else if(is.character(x) || is.character(table))
        index <- .semijoin(as.character(x), as.character(table))

    else if(is.raw(x) && is.raw(table))
        index <- .semijoin(x, table)
    else if(is.raw(x) || is.raw(table))
        index <- .semijoin(as.character(x), as.character(table))

    else if(is.double(x) || is.double(table))
        index <- .semijoin(as.double(x), as.double(table))
    else if(is.integer(x) || is.integer(table))
        index <- .semijoin(as.integer(x), as.integer(table))
    else if(is.logical(x) || is.logical(table))
        index <- .semijoin(as.logical(x), as.logical(table))

    else
        stop("'match' requires vector arguments")
}

match <- function(x, table, nomatch, incomparables) {
    index <- .match(x, table)
    notcompared <- .match(x, incomparables)
    index[is.na(index) | !is.na(notcompared)] <- as.integer(nomatch)
    index
}

