
.mode <- function(m) {
    switch(m,
        'numeric' = c('integer', 'double'),
        'name' = 'symbol',
        m)
}

get <- function(x, envir, mode, inherits) {
    mode <- .mode(mode)
    if (.env_exists(envir, x) 
        && (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode == "any"))
        return(envir[[x]])
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- environment(envir)
            if (.env_exists(envir, x) 
                && (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode == "any"))
                return(envir[[x]])
        }
    }
    stop('object not found')
}

