
call <- function(name, ...) {
    if(is.function(name))
        r <- list(name, ...)
    else
        r <- list(as.symbol(name), ...)
    class(r) <- 'call'
    r
}

is.call <- function(x) {
    any('call' == class(x))
}

as.call <- function(x) {
    if(!is.list(x))
        stop("invalid argument list")
    if(length(x) == 0)
        stop("invalid length 0 argument")
    class(x) <- 'call'
    x
}
