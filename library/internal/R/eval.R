
eval <- function(expr, envir, enclos) {
    if (is.list(envir) || is.pairlist(envir)) {
        envir <- as.environment(envir)
        environment(envir) <- enclos
    }

    promise('f', expr, envir, core::parent.frame(0))
    f
}

