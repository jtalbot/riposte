
file.info <- function(files) {
    r <- .Map('fileinfo_map', list(files), c('double', 'logical', 'integer', 'integer', 'integer', 'integer', 'integer', 'integer', 'character', 'character'))

    class(r[[3]]) <- 'octmode'

    names(r) <- characters('size', 'isdir', 'mode', 'mtime', 'ctime', 'atime', 'uid', 'gid', 'uname', 'grname')

    r
}

Sys.glob <- function(paths, dirmark) {
    .External(sysglob(as.character(paths), as.logical(dirmark)))
}

normalizedPath <- function(path, winslash, mustWork) {
    r <- .Map('realpath_map', list(as.character(path), as.character(winslash)), c('character'))[[1]]
    
    mustWork <- as.logical(mustWork)
    if(identical(mustWork, TRUE) && any(is.na(r)))
        stop("No such file or directory")
    else if(is.na(mustWork) && any(is.na(r)))
        warning("No such file or directory")

    r
}
