
.mode <- function(m) {
    switch(m,
        'numeric' = c('integer', 'double'),
        'name' = 'symbol',
        m)
}

get <- function(x, envir, mode, inherits) {
    mode <- .mode(mode)
    
    if (!is.nil(.get(envir, x)) 
        && (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode == "any")) {
        return(envir[[x]])
    }
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- .getenv(envir)
            if (!is.nil(.get(envir, x))
                && (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode == "any"))
                return(envir[[x]])
        }
    }
    .stop('object not found')
}

mget <- function(x, envir, mode, ifnotfound, inherits) {
    r <- vector("list", length(x))

    for(i in seq_len(length(x))) {
        r[[i]] <- get(x[[i]], envir, mode, inherits)
        if (!is.nil(.get(envir, x[[i]]))
            && (any(match(typeof(envir[[x[[i]]]]), mode, 0, NULL)) || mode == "any")) {
            r[[i]] <- envir[[x[[i]]]]
            next
        }
        if (inherits) {
            while(envir != emptyenv()) {
                envir <- environment(envir)
                if (!is.nil(.get(envir, x[[i]]))
                    && (mode(envir[[x[[i]]]]) == mode || mode == "any")) {
                    r[[i]] <- envir[[x[[i]]]]
                    next
                }
            }
        }
        r[[i]] <- ifnotfound[[i]]
    }

    names(r) <- x
    r
}

