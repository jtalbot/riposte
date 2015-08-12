
ls <- function(envir, all.names, sorted) {
    n <- .env_names(envir)
    if (all.names)
        n
    else
        n[substr(n, 1L, 1L) != '.']
}

