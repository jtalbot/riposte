
Sys.glob <- function(paths, dirmark) {
    .External(sysglob(as.character(paths), as.logical(dirmark)))
}

