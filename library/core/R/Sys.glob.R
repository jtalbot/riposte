
Sys.glob <- function(paths, dirmark) {
    .Riposte('sysglob', as.character(paths), as.logical(dirmark))
}

