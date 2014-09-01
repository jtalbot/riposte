
eval <- function(expr, envir, enclos) {
    if (is.list(envir) || is.pairlist(envir)) {
        envir <- as.environment(envir)
        environment(envir) <- enclos
    }

    if(is.expression(expr)) {
        f <- NULL
        for(e in expr) {
            promise('f', e, envir, .getenv(NULL))
            f
        }
    }
    else { 
        promise('f', expr, envir, .getenv(NULL))
        f
    }
}

