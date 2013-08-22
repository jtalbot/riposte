
dimnames <- function(x) UseMethod('dimnames', x)

dimnames.default <- function(x) {
    if(length(attr(x, 'dim')) == 1L)
        attr(x, 'names')
    else
        attr(x, 'dimnames')
}


`dimnames<-` <- function(x, value) UseMethod('dimnames<-', x)

`dimnames<-.default` <- function(x, value) {
    if(length(value) == 0L)
        value <- NULL

    if(is.null(value)) {
        attr(x, 'dimnames') <- NULL
    }
    else {
        if(!is.list(value))
            .stop("'dimnames' must be a list")
        
        d <- attr(x, 'dim')
        if(length(value) != length(d))
            .stop(sprintf("length of 'dimnames' [%d] not equal to array extent", length(value)))

        value <- .Map(function(x) if(length(x) == 0L) NULL else x, list(value))
        
        for(i in seq_len(length(value))) {
            if(!is.null(value[[i]]) && length(value[[i]]) != d[[i]])
                .stop(sprintf("length of 'dimnames' [%d] not equal to array extent", length(value)))
        }

        if(length(d) == 1L)
            attr(x, 'names') <- value
        else
            attr(x, 'dimnames') <- value
    }
    x
}
