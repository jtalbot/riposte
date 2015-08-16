
drop <- function(x) {
    d <- dim(x)
    nd <- d[d!=1L]

    if(!is.null(attr(x,'dimnames')))
    {
        if(length(nd)===0L) {
            attr(x,'dimnames') <- NULL
        }
        else if(length(nd)===1L) {
            attr(x,'names') <- attr(x,'dimnames')[d!=1L][[1]]
            attr(x,'dimnames') <- NULL
        }
        else {
            attr(x,'dimnames') <- attr(x,'dimnames')[d!=1L]
        }
    }
    
    if(length(nd) === 1L || length(nd) === 0L)
        attr(x,'dim') <- NULL

    x    
}

