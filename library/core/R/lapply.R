
lapply <- function(X, FUN) {
    # annoyingly, R doesn't pass the dots here, have to go back up and get them
    promise('MoreArgs', quote(list(...)), .frame(1L), .getenv(NULL))
    # enlist the MoreArgs so they get repeated correctly...
    MoreArgs <- .Map(function(x) list(x), list(MoreArgs))
    args <- list(X)
    args[seq_len(length(MoreArgs))+1L] <- MoreArgs

    if(!is.null(.frame(1L)[['__names__']])) {
        n <- ''
        n[seq_len(length(MoreArgs))+1L] <- .frame(1L)[['__names__']]
        names(args) <- n
    }

    r <- .Map(FUN, args)
    names(r) <- names(X)
    r
}

vapply <- function(X, FUN, FUN.VALUE, USE.NAMES) {
    # annoyingly, R doesn't pass the dots here, have to go back up and get them
    promise('MoreArgs', quote(list(...)), .frame(1L), .getenv(NULL))
    # enlist the MoreArgs so they get repeated correctly...
    MoreArgs <- .Map(function(x) list(x), list(MoreArgs))

    args <- list(X)
    args[seq_len(length(MoreArgs))+1L] <- MoreArgs

    if(!is.null(.frame(1L)[['__names__']])) {
        n <- ''
        n[seq_len(length(MoreArgs))+1L] <- .frame(1L)[['__names__']]
        names(args) <- n
    }

    types <- .Map(function(x) .type(x), list(FUN.VALUE), 'character')[[1L]]
    r <- unlist(.Map(FUN, args, types), FALSE, FALSE)
    if(length(types) > 0) {
        dim(r) <- c(length(r)/length(types), length(types))
        r <- t.default(r)
    }
    
    n <- names(X)
    if(!is.null(X))
        names(r) <- n
    else if(USE.NAMES && is.character(X))
        names(r) <- X
    r
}

