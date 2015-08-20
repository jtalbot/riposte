
.mode <- function(m) {
    switch(m,
        'numeric' = c('integer', 'double'),
        'name' = 'symbol',
        'function' = 'closure',
        m)
}

get <- function(x, envir, mode, inherits) {
    mode <- .mode(mode)

    if (.env_has(envir, x)
        && (mode == 'any' || any(match(typeof(envir[[x]]), mode, 0, NULL)))) {
        return(envir[[x]])
    }
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- .getenv(envir)
            if (.env_has(envir, x)
                && (mode == 'any' ||
                    any(match(typeof(envir[[x]]), mode, 0, NULL))))
                return(envir[[x]])
        }
    }
    .stop(sprintf("object '%s' not found",x))
}

mget <- function(x, envir, mode, ifnotfound, inherits) {
    r <- vector("list", length(x))

    for(i in seq_len(length(x))) {
        r[[i]] <- get(x[[i]], envir, mode, inherits)
        if (.env_has(envir, x[[i]])
            && (mode == 'any' || any(match(typeof(envir[[x[[i]]]]), mode, 0, NULL)))) {
            r[[i]] <- envir[[x[[i]]]]
            next
        }
        if (inherits) {
            while(envir != emptyenv()) {
                envir <- environment(envir)
                if (.env_has(envir, x[[i]])
                    && (mode == 'any' || mode(envir[[x[[i]]]]) == mode)) {
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

