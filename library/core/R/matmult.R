
`%*%` <- function(x, y) {
    xd <- dim(x)
    yd <- dim(y)
    if(length(xd) != 2L) xd <- NULL
    if(length(yd) != 2L) yd <- NULL

    if(is.null(xd) && is.null(yd)) {
        xd <- c(1L, length(x))
        yd <- c(length(y), 1L)
    }
    else if(is.null(xd)) {
        if(yd[[1L]] == length(x))
            xd <- c(1L, length(x))
        else
            xd <- c(length(x), 1L)
    }
    else if(is.null(yd)) {
        if(xd[[1L]] == length(y))
            yd <- c(1L, length(y))
        else
            yd <- c(length(y), 1L)
    }

    if(xd[[2L]] != yd[[1L]])
        .stop("Matrices are not conformable")

    if(xd[[1L]] == 1L && yd[[2L]] == 1L) {
        return(sum(strip(x)*strip(y)))
    } 
    #else if(xd[[1L]] == 1L) {
    #    r <- double(0)
    #    for(i in 1L:yd[[2L]])
    #        r[[i]] <- sum(strip(x)*strip(y[,i]))
    #    return(r)
    #}
    #else if(yd[[2L]] == 1L) {
    #    r <- 0
    #    for(i in 1L:xd[[2L]]) {
    #        r <- r + strip(x[,i])*strip(y)[[i]]
    #    }
    #    return(r)
    #}
    else {
        r <- .Riposte('matrixmultiply',strip(x),xd[[1L]],xd[[2L]],y,yd[[1L]],yd[[2L]])
        dim(r) <- c(xd[[1L]],yd[[2L]])
        return(r)
    }
}

