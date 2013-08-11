
.match <- function(x, table) {
    if(is.list(x) || is.list(table))
        .semijoin(as.character.default(x), as.character.default(table))
    else if(is.character.default(x) || is.character.default(table))
        .semijoin(as.character.default(x), as.character.default(table))

    else if(is.raw(x) && is.raw(table))
        .semijoin(x, table)
    else if(is.raw(x) || is.raw(table))
        .semijoin(as.character.default(x), as.character.default(table))

    else if(is.double(x) || is.double(table))
        .semijoin(as.double(x), as.double(table))
    else if(is.integer(x) || is.integer(table))
        .semijoin(as.integer(x), as.integer(table))
    else if(is.logical(x) || is.logical(table))
        .semijoin(as.logical(x), as.logical(table))
    else if(is.null(x))
        vector('integer',0L)
    else if(is.null(table))
        vector(0L, length(x))
    else
        .stop("'match' requires vector arguments")
}

match <- function(x, table, nomatch, incomparables) {
    if(identical(incomparables, FALSE))
        incomparables <- NULL

    index <- .match(x, table)
    notcompared <- .match(x, incomparables)
    index[index==0L | notcompared != 0L] <- as.integer(nomatch)
    index
}

