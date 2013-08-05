
drop <- function(x) {
    d <- dim(x)
    nd <- d[d!=1L]

    if(length(nd) == 1L || length(nd) == 0L)
        attr(x,'dim') <- NULL

    if(!is.null(dimnames(x))) {
        if(length(nd)==0L) {
            attr(x,'dimnames') <- NULL
            attr(x,'names') <- dimnames(x)[d==1L][[1]]
        }
        else if(length(nd)==1L) {
            attr(x,'dimnames') <- NULL
            attr(x,'names') <- dimnames(x)[d!=1L]
        }
        else {
            attr(x,'dimnames') <- dimnames(x)[d!=1L]
        }
    }
    x    
}

