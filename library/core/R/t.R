
t.default <- function(x) {
    d <- dim(x)
    if(is.null(d) || length(d) == 1L) {
        attr(x, 'dim') <- c(1L, length(x))
        x
    }
    else if(length(d) == 2L) {
        attr(x, 'dim') <- c(d[[2L]], d[[1L]])
        x
    }
    else {
        .stop('argument is not a matrix')
    }
}
