
ls <- function(envir, all.names) {
    n <- names(as.list(envir))
    if (all.names)
        n
    else
        n[substr(n, 1L, 1L) != '.']
}

