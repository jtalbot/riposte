
exists <- function(x, envir, mode, inherits) {
    mode <- .mode(mode)                 # defined in get.R

    if (!is.nil(.get(envir, x))
        && (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode == "any"))
        return(TRUE)
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- .getenv(envir)
            if (!is.nil(.get(envir, x))
                && (any(match(typeof(envir[[x]]), mode, 0, NULL)) || mode == "any"))
                return(TRUE)
        }
    }
    return(FALSE)
}

