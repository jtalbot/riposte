
assign <- function(x, value, envir, inherits) {
    if(!inherits || .env_exists(envir, x)) {
        envir[[x]] <- value
        return(value)
    }
    else {
        while(envir != emptyenv()) {
            envir <- environment(envir)
            if(.env_exists(envir, value)) {
                envir[[x]] <- value
                return(value)
            }
        }
    }
    g <- globalenv()
    g[[x]] <- value
    return(value)
}
