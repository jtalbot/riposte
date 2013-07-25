
which <- function(x) {
    if(!is.logical(x))
        .stop("argument to 'which' is not logical")

    seq_len(length(x))[x]
}

