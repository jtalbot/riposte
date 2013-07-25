
file.info <- function(files) {
    r <- .Map('fileinfo_map', list(files), 
        .characters('double', 'logical', 'integer', 'integer', 'integer', 'integer', 'integer', 'integer', 'character', 'character'))

    class(r[[3]]) <- 'octmode'

    names(r) <- .characters('size', 'isdir', 'mode', 'mtime', 'ctime', 'atime', 'uid', 'gid', 'uname', 'grname')

    r
}

