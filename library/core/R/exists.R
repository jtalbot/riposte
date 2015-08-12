
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

get0 <- function(x, envir, mode, inherits, ifnotfound)
{
    mode <- .mode(mode)                 # defined in get.R

    val <- .get(envir, x)

    if (!is.nil(val)
        && (any(match(typeof(val), mode, 0, NULL)) || mode == "any"))
        return(val)
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- .getenv(envir)
            val <- .get(envir, x)
            if (!is.nil(val)
                && (any(match(typeof(val), mode, 0, NULL)) || mode == "any"))
                return(val)
        }
    }
    return(ifnotfound)
}

