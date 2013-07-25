
eval <- function(expr, envir, enclos) {
    if (is.list(envir) || is.pairlist(envir)) {
        envir <- as.environment(envir)
        environment(envir) <- enclos
    }

    promise('f', expr, envir, .getenv(NULL))
    f
}

