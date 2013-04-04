
get <- function(x, envir, mode, inherits) {
    if (.env_exists(envir, x) && (mode(envir[[x]]) == mode || mode == "any"))
        return(envir[[x]])
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- environment(envir)
            if (.env_exists(envir, x) && (mode(envir[[x]]) == mode || mode == "any"))
                return(envir[[x]])
        }
    }
    stop('object not found')
}

mget <- function(x, envir, mode, ifnotfound, inherits) {
    r <- vector("list", length(x))

    # TODO -- deal with ifnotfound

    for(i in seq_along(x)) {
        if (.env_exists(envir, x[[i]]) && (mode(envir[[x[[i]]]]) == mode || mode == "any"))
            r[[i]] <- envir[[x[[i]]]]
        if (inherits) {
            while(envir != emptyenv()) {
                envir <- environment(envir)
                if (.env_exists(envir, x[[i]]) && (mode(envir[[x[[i]]]]) == mode || mode == "any"))
                    r[[i]] <- envir[[x[[i]]]]
            }
        }
    }

    names(r) <- x
    r
}

