
mapply <- function(FUN, dots, MoreArgs)
{
    # enlist the MoreArgs so they get repeated correctly...
    MoreArgs <- .Map(function(x) list(x), list(MoreArgs))

    dots[seq_len(length(MoreArgs))+length(dots)] <- MoreArgs

    #if(!is.null(.frame(1L)[['.__names__.']])) {
    #    n <- ''
    #    n[seq_len(length(MoreArgs))+1L] <- .frame(1L)[['.__names__.']]
    #    names(args) <- n
    #}

    .Map(FUN, dots)
}

