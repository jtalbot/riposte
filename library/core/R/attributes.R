
attributes <- function(obj) {
    attributes(obj)
}

`attributes<-` <- function(obj, value) {
    obj <- strip(obj)
    
    if(!is.null(value)) {
        n <- attr(value, 'names')

        if(is.null(n))
            .stop('attributes must be named')

        dim <- .semijoin('dim', n)
        if(dim != 0L)
            dim(obj) <- value[[dim]]

        for(i in seq_len(length(value))) {
            if(n[[i]] == 'dimnames')
                dimnames(obj) <- value[[i]]
            else if(n[[i]] == 'names')
                names(obj) <- value[[i]]
            else if(n[[i]] == 'class')
                class(obj) <- value[[i]]
            else if(n[[i]] != 'dim')
                attr(obj, n[[i]]) <- value[[i]]
        }
    }
    obj
}

