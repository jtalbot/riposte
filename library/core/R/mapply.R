
mapply <- function(FUN, dots, MoreArgs) {

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

