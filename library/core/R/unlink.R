
unlink <- function(x, recursive, force) {
    x <- Sys.glob(as.character(x), FALSE)
    .Riposte('sysunlink', x, as.logical(recursive), as.logical(force))
}

