
file.info <- function(files, extra_cols) {
    r <- .Map('fileinfo_map', list(files), 
        .characters('double', 'logical', 'integer', 'integer', 'integer', 'integer', 'integer', 'integer', 'character', 'character'))

    class(r[[3]]) <- 'octmode'

    names(r) <- .characters('size', 'isdir', 'mode', 'mtime', 'ctime', 'atime', 'uid', 'gid', 'uname', 'grname')

    r
}

