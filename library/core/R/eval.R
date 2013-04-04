
eval <- function(expr, envir, enclos) {
    if (is.list(envir) || is.pairlist(envir)) {
        envir <- as.environment(envir)
        environment(envir) <- enclos
    }

    f <- promise(expr, envir)
    f
}

