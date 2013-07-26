
Sys.getenv <- function(x, unset) {
    if(length(x) == 0L) {
        .External(sysgetenv())
    }
    else {
        .Map('sysgetenv_map', list(x, unset), 'character')[[1]]
    }
}

