
call <- function(name, ...) {
    if(is.function(name))
        r <- list(name, ...)
    else
        r <- list(as.symbol(name), ...)
    attr(r, 'class') <- 'call'
    r
}

is.call <- function(x) {
    any('call' == attr(x, 'class'))
}

as.call <- function(x) {
    if(!is.list(x))
        .stop("invalid argument list")
    if(length(x) == 0)
        .stop("invalid length 0 argument")
    attr(x, 'class') <- 'call'
    x
}
