
as.function.default <- function(x, envir) {
    call <- list()
    call[[1]] <- 'function'
    call[[2]] <- x[-length(x)]
    call[[3]] <- x[[length(x)]]
    call[[4]] <- deparse(call[[3]])
    class(call) <- 'call'
    promise('p', call, envir, .getenv(NULL))
    return(p)
}

