
assign <- function(x, value, envir, inherits) {
    if(!inherits || !is.nil(.get(envir, x))) {
        envir[[x]] <- value
        return(value)
    }
    else {
        while(envir != emptyenv()) {
            envir <- .getenv(envir)
            if(!is.nil(.get(envir, value))) {
                envir[[x]] <- value
                return(value)
            }
        }
    }
    g <- .env_global()
    g[[x]] <- value
    return(value)
}

