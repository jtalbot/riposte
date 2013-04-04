
exists <- function(x, envir, mode, inherits) {
    if (.env_exists(envir, x) && (mode(envir[[x]]) == mode || mode == "any"))
        return(TRUE)
    if (inherits) {
        while(envir != emptyenv()) {
            envir <- environment(envir)
            if (.env_exists(envir, x) && (mode(envir[[x]]) == mode || mode == "any"))
                return(TRUE)
        }
    }
    return(FALSE)
}

